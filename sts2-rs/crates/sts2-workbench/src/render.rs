//! Human-readable rendering of card text and chosen actions.
//!
//! Port of `sts2-pycli/python/workbench/render.py`.
//!
//! The game ships card descriptions as templates (`"Deal {Damage:diff()}
//! damage."`) plus a per-card `stats` dict with the resolved numbers, and logs
//! actions as `{cmd, action, args}` dicts referencing cards by index. This
//! module turns both into plain English for display.
//!
//! Every string produced here must be byte-identical to the Python module's
//! output (the golden-JSON parity tests diff the two), so the port follows the
//! Python control flow line for line and reproduces Python's `str()` /
//! truthiness / `dict.get` semantics through the [`py`] helpers below, which
//! the sibling modules (`options`, `statediff`, `combatlog`, `playlog`) share.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

/// Sentinel verb for a step taken by the NATIVE run agent (run-mode autoplay).
/// Its `args` are a raw Rust `RunAction` dict rather than workbench-verb
/// args. Defined here (not in the branching module, which imports this one)
/// so both the recorder and the renderer share one spelling.
pub const NATIVE_ACTION: &str = "__agent_native__";

/// Native RunAction / logged-native kinds → the C# verb vocabulary the
/// branches below speak. Only the combat kinds `_native_action` rewrites need
/// an entry; everything else already round-trips as its own kind.
const NATIVE_KIND_VERB: &[(&str, &str)] = &[("play", "play_card"), ("drink_potion", "use_potion")];

/// Icon / prefix tokens that look like `{Var}` substitutions but never come
/// from a backend's `stats` dict — [`resolve_card_text`] handles them inline.
const PSEUDO_VARS: &[&str] = &["energyprefix", "singlestaricon"];
const STAR_PSEUDO_VARS: &[&str] = &["singlestaricon"];

// ── Python-semantics helpers ─────────────────────────────────────────────────

/// Reproductions of the Python built-ins the workbench modules lean on
/// (`str()`, truthiness, `dict.get`, `int()`, `==`, `str.strip`) over
/// `serde_json::Value`, so a port can follow the Python line for line and
/// still produce byte-identical text.
///
/// Known, deliberate gaps (each would be a `TypeError` / unusual input on the
/// Python side, not a difference in normal output):
///
/// * `str()` of a list / dict renders serde's compact JSON, not Python's
///   `repr` (`['a']` vs `["a"]`). No workbench template or label embeds one.
/// * `int()` of a string accepts only an optionally-signed ASCII integer
///   after trimming (Python also takes `_` separators and Unicode digits).
/// * `\s` / `str.isspace()` is approximated by Unicode `White_Space` plus
///   U+001C..=U+001F, which Python's `isspace` accepts and Unicode does not.
pub(crate) mod py {
    use serde_json::Value;

    /// Python truthiness of a JSON value.
    pub(crate) fn truthy(v: &Value) -> bool {
        match v {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        }
    }

    /// Truthiness of an optional value (`None` is falsy).
    pub(crate) fn truthy_opt(v: Option<&Value>) -> bool {
        v.is_some_and(truthy)
    }

    /// `d.get(k)` — `None` for a missing key AND for a present `null`, which is
    /// what every Python `is None` / truthiness check downstream sees.
    pub(crate) fn get<'a>(d: &'a Value, k: &str) -> Option<&'a Value> {
        match d.get(k) {
            Some(Value::Null) | None => None,
            some => some,
        }
    }

    /// `k in d` / `d.get(k, default)` — the raw slot, `Some(Null)` for a key
    /// that is present but `null` (Python's `.get(k, default)` returns `None`
    /// there, NOT the default).
    pub(crate) fn get_raw<'a>(d: &'a Value, k: &str) -> Option<&'a Value> {
        d.get(k)
    }

    /// `d.get(k, default)`.
    pub(crate) fn get_or<'a>(d: &'a Value, k: &str, default: &'a Value) -> &'a Value {
        d.get(k).unwrap_or(default)
    }

    /// `d.get(k) or []` as a slice — an absent / `null` / empty / non-array
    /// slot reads as no items.
    pub(crate) fn get_list<'a>(d: &'a Value, k: &str) -> &'a [Value] {
        list(d.get(k))
    }

    /// `v or []` as a slice.
    pub(crate) fn list(v: Option<&Value>) -> &[Value] {
        match v {
            Some(Value::Array(a)) => a,
            _ => &[],
        }
    }

    /// `d.get(k) or {}` — the object under `k`, or an empty one.
    pub(crate) fn get_obj<'a>(d: &'a Value, k: &str) -> &'a Value {
        static EMPTY: Value = Value::Null;
        match d.get(k) {
            Some(v @ Value::Object(_)) => v,
            _ => &EMPTY,
        }
    }

    /// `str(v)` for the scalar shapes the workbench prints.
    pub(crate) fn to_str(v: &Value) -> String {
        match v {
            Value::Null => "None".to_string(),
            Value::Bool(true) => "True".to_string(),
            Value::Bool(false) => "False".to_string(),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    i.to_string()
                } else if let Some(u) = n.as_u64() {
                    u.to_string()
                } else {
                    float_repr(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }

    /// `str(v)` where `v` may be absent (`str(None)` → `"None"`).
    pub(crate) fn opt_str(v: Option<&Value>) -> String {
        v.map_or_else(|| "None".to_string(), to_str)
    }

    /// Python's `repr(float)`: shortest round-trip digits, fixed notation for
    /// exponents in `-4..16`, else `d.ddde±XX`; integral values keep a `.0`.
    pub(crate) fn float_repr(f: f64) -> String {
        if f.is_nan() {
            return "nan".to_string();
        }
        if f.is_infinite() {
            return if f < 0.0 { "-inf" } else { "inf" }.to_string();
        }
        if f == 0.0 {
            return if f.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
        }
        let sci = format!("{:e}", f.abs()); // "2.5e0", "1e20", "1.5e-7"
        let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
        let exp: i32 = exp.parse().unwrap_or(0);
        let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
        let mut out = String::new();
        if f < 0.0 {
            out.push('-');
        }
        if (-4..16).contains(&exp) {
            if exp >= 0 {
                let int_len = exp as usize + 1;
                if digits.len() <= int_len {
                    out.push_str(&digits);
                    out.extend(std::iter::repeat_n('0', int_len - digits.len()));
                    out.push_str(".0");
                } else {
                    out.push_str(&digits[..int_len]);
                    out.push('.');
                    out.push_str(&digits[int_len..]);
                }
            } else {
                out.push_str("0.");
                out.extend(std::iter::repeat_n('0', (-exp - 1) as usize));
                out.push_str(&digits);
            }
        } else {
            out.push_str(&digits[..1]);
            if digits.len() > 1 {
                out.push('.');
                out.push_str(&digits[1..]);
            }
            out.push('e');
            out.push(if exp < 0 { '-' } else { '+' });
            out.push_str(&format!("{:02}", exp.abs()));
        }
        out
    }

    /// A Python number: `bool` is an `int` subclass, ints are exact, and
    /// anything mixed with a float is a float.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub(crate) enum Num {
        Int(i64),
        Float(f64),
    }

    impl Num {
        pub(crate) fn as_f64(self) -> f64 {
            match self {
                Num::Int(i) => i as f64,
                Num::Float(f) => f,
            }
        }

        /// `a - b`.
        pub(crate) fn sub(self, other: Num) -> Num {
            match (self, other) {
                (Num::Int(a), Num::Int(b)) => a
                    .checked_sub(b)
                    .map_or(Num::Float(a as f64 - b as f64), Num::Int),
                (a, b) => Num::Float(a.as_f64() - b.as_f64()),
            }
        }

        /// `abs(n)`.
        pub(crate) fn abs(self) -> Num {
            match self {
                Num::Int(i) => i
                    .checked_abs()
                    .map_or(Num::Float((i as f64).abs()), Num::Int),
                Num::Float(f) => Num::Float(f.abs()),
            }
        }

        /// `n < 0`.
        pub(crate) fn is_negative(self) -> bool {
            match self {
                Num::Int(i) => i < 0,
                Num::Float(f) => f < 0.0,
            }
        }

        /// `n > 0`.
        pub(crate) fn is_positive(self) -> bool {
            match self {
                Num::Int(i) => i > 0,
                Num::Float(f) => f > 0.0,
            }
        }

        /// `str(n)`.
        pub(crate) fn to_str(self) -> String {
            match self {
                Num::Int(i) => i.to_string(),
                Num::Float(f) => float_repr(f),
            }
        }
    }

    /// `isinstance(v, (int, float))` → the number (bools count as ints).
    pub(crate) fn num(v: &Value) -> Option<Num> {
        match v {
            Value::Bool(b) => Some(Num::Int(i64::from(*b))),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(Num::Int(i))
                } else {
                    n.as_f64().map(Num::Float)
                }
            }
            _ => None,
        }
    }

    /// `isinstance(v, int)` → the int (bools are ints; floats are not).
    pub(crate) fn as_int(v: &Value) -> Option<i64> {
        match v {
            Value::Bool(b) => Some(i64::from(*b)),
            Value::Number(n) => n.as_i64(),
            _ => None,
        }
    }

    /// `str(int(v))` — bools become `0`/`1`, floats truncate toward zero,
    /// strings parse as a trimmed signed integer. `None` where Python raises.
    pub(crate) fn int_str(v: &Value) -> Option<String> {
        match v {
            Value::Bool(b) => Some(i64::from(*b).to_string()),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(i.to_string())
                } else if let Some(u) = n.as_u64() {
                    Some(u.to_string())
                } else {
                    let f = n.as_f64()?;
                    if !f.is_finite() {
                        return None;
                    }
                    let t = f.trunc();
                    Some(if t == 0.0 {
                        "0".to_string()
                    } else {
                        format!("{t:.0}")
                    })
                }
            }
            Value::String(s) => strip(s).parse::<i64>().ok().map(|i| i.to_string()),
            _ => None,
        }
    }

    /// `int(v or 0)` — the engine's amount fields (`unblocked`, `amount`).
    pub(crate) fn int_or_zero(v: Option<&Value>) -> i64 {
        match v {
            Some(v) if truthy(v) => match v {
                Value::Bool(_) => 1,
                Value::Number(n) => n
                    .as_i64()
                    .or_else(|| n.as_f64().map(|f| f.trunc() as i64))
                    .unwrap_or(0),
                Value::String(s) => strip(s).parse::<i64>().unwrap_or(0),
                _ => 0,
            },
            _ => 0,
        }
    }

    /// Python `==` across the JSON scalars (`1 == 1.0 == True`, `None == None`).
    pub(crate) fn eq(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::String(x), Value::String(y)) => x == y,
            (Value::Array(x), Value::Array(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| eq(p, q))
            }
            (Value::Object(x), Value::Object(y)) => {
                x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| eq(v, w)))
            }
            _ => match (num(a), num(b)) {
                (Some(Num::Int(x)), Some(Num::Int(y))) => x == y,
                (Some(x), Some(y)) => x.as_f64() == y.as_f64(),
                _ => false,
            },
        }
    }

    /// `str.isspace()` for one char — see the module note on the approximation.
    pub(crate) fn is_space(c: char) -> bool {
        c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
    }

    /// `s.strip()`.
    pub(crate) fn strip(s: &str) -> &str {
        s.trim_matches(is_space)
    }
}

// ── Template DSL ─────────────────────────────────────────────────────────────

/// Split `text` on `sep`, ignoring separators inside `{...}`.
#[must_use]
pub fn split_top_level(text: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth: usize = 0;
    let mut start = 0;
    for (i, ch) in text.char_indices() {
        if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth = depth.saturating_sub(1);
        } else if ch == sep && depth == 0 {
            parts.push(&text[start..i]);
            start = i + ch.len_utf8();
        }
    }
    parts.push(&text[start..]);
    parts
}

/// Byte index of the `}` matching the `{` at `open`, or `None` when the text
/// runs out first (unbalanced).
fn matching_close(text: &str, open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut j = open;
    while j < bytes.len() {
        match bytes[j] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(j);
                }
            }
            _ => {}
        }
        j += 1;
    }
    None
}

/// Recursively resolve `{...}` directives in `text` (balanced braces).
///
/// Handles three directive shapes (branches may nest further `{...}`):
/// `{Cond:then|else}` boolean wrappers (`InCombat` / `IfUpgraded`, the latter
/// with an optional `show:` prefix), `{Var:plural:singular|plural}` selection,
/// and `{Var:...}` value substitution from `stats`.
fn render_template(
    text: &str,
    stats: &Map<String, Value>,
    flags: &Map<String, Value>,
    markup: bool,
) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            // Copy the whole non-brace run at once (chars, not bytes, so a
            // multi-byte glyph is never split).
            let next = text[i..].find('{').map_or(text.len(), |k| i + k);
            out.push_str(&text[i..next]);
            i = next;
            continue;
        }
        // Find the matching close brace for this open brace.
        let Some(j) = matching_close(text, i) else {
            // Unbalanced — emit the rest verbatim.
            out.push_str(&text[i..]);
            break;
        };
        out.push_str(&eval_directive(&text[i + 1..j], stats, flags, markup));
        i = j + 1;
    }
    out
}

/// Resolve a `choose(opt1|opt2|...):branch1|branch2|...` directive body.
///
/// `value` (a card's morph var, e.g. CardType "Skill") selects the option
/// index in the `choose(...)` list, then the parallel branch. Returns the raw
/// chosen branch (caller renders it), or `None` if `rest` isn't a choose
/// directive. Missing/unknown `value` falls back to the first branch.
#[must_use]
pub fn choose_branch<'a>(rest: &'a str, value: Option<&Value>) -> Option<&'a str> {
    let rest_after = rest.strip_prefix("choose(")?;
    let sep = rest.find("):")?;
    let options: Vec<String> = rest_after[..sep - "choose(".len()]
        .split('|')
        .map(|o| py::strip(o).to_lowercase())
        .collect();
    let branches = split_top_level(&rest[sep + 2..], '|');
    let idx = match value {
        Some(v) if !v.is_null() => {
            let needle = py::to_str(v).to_lowercase();
            options.iter().position(|o| *o == needle).unwrap_or(0)
        }
        _ => 0,
    };
    Some(branches.get(idx).copied().unwrap_or(""))
}

/// `energyIcons()` / `starIcons(2)` — an inline run of cost glyphs, with the
/// count either read from `stats` (empty parens) or stated literally.
///
/// Hand-rolled `^(energyIcons|starIcons)\((\d*)\)$` (case-insensitive): returns
/// `(is_star, literal_digits)`.
fn icon_fmt(rest: &str) -> Option<(bool, &str)> {
    let s = py::strip(rest);
    let inner = s.strip_suffix(')')?;
    let (is_star, digits) = if inner.len() >= "energyicons(".len()
        && inner.as_bytes()[.."energyicons(".len()].eq_ignore_ascii_case(b"energyicons(")
    {
        (false, &inner["energyicons(".len()..])
    } else if inner.len() >= "staricons(".len()
        && inner.as_bytes()[.."staricons(".len()].eq_ignore_ascii_case(b"staricons(")
    {
        (true, &inner["staricons(".len()..])
    } else {
        return None;
    };
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((is_star, digits))
}

/// The count an icon directive renders: a literal from `energyIcons(1)`, or
/// whatever `stats` holds for the var (possibly nothing).
enum IconCount<'a> {
    Literal(&'a str),
    Stat(Option<&'a Value>),
}

impl IconCount<'_> {
    /// `isinstance(count, (int, float))` — the literal always is; `int(count)`.
    fn int_str(&self) -> Option<String> {
        match self {
            IconCount::Literal(digits) => Some(digits.parse::<i64>().map_or_else(
                |_| digits.trim_start_matches('0').to_string(),
                |i| i.to_string(),
            )),
            IconCount::Stat(v) => v.and_then(py::int_str),
        }
    }
}

fn eval_directive(
    body: &str,
    stats: &Map<String, Value>,
    flags: &Map<String, Value>,
    markup: bool,
) -> String {
    let (head, rest) = body.split_once(':').unwrap_or((body, ""));
    let key = py::strip(head).to_lowercase();

    // `{Var:choose(a|b|c):branchA|branchB|branchC}` — pick a branch by the
    // card's morph value (Mad Science's CardType). The value may travel in the
    // morph `flags` or, for callers that stash it there, in `stats`.
    let chooser = if flags.contains_key(&key) {
        flags.get(&key)
    } else {
        stats.get(&key)
    };
    if let Some(chosen) = choose_branch(rest, chooser) {
        return render_template(chosen, stats, flags, markup);
    }

    // Boolean `{Cond:then|else}` wrappers: `InCombat` / `IfUpgraded` plus any
    // per-card morph flag the caller supplied (Mad Science's HasRider and
    // rider names). A flag is a condition iff it's present in `flags`.
    if let Some(flag) = flags.get(&key) {
        let branches = rest.strip_prefix("show:").unwrap_or(rest);
        // Split on the top-level `|` (branches may contain nested braces).
        let halves = split_top_level(branches, '|');
        let then_text = halves[0];
        let else_text = halves.get(1).copied().unwrap_or("");
        let chosen = if py::truthy(flag) {
            then_text
        } else {
            else_text
        };
        return render_template(chosen, stats, flags, markup);
    }

    // Cost glyphs. The game draws these as little energy orbs / stars inline —
    // "Gain ●●●", "Add a 0● copy", "Whenever you spend ★" — in four shapes:
    //   {Energy:energyIcons()}        N from stats
    //   {energyPrefix:energyIcons(1)} N given literally, no stat
    //   {Stars:starIcons()}           the Regent's star equivalent of the first
    //   {singleStarIcon}              exactly one star, no formatter at all
    // In markup mode they become `[energy]N[/energy]` / `[star]N[/star]` for
    // the card skin to draw. Plain mode keeps its existing output — a bare
    // number for the stat forms, nothing for the literal ones — because action
    // labels and logs read that text.
    let icon = icon_fmt(rest);
    let is_pseudo = PSEUDO_VARS.contains(&key.as_str());
    if icon.is_some() || is_pseudo {
        let mut is_star = STAR_PSEUDO_VARS.contains(&key.as_str());
        let mut count = IconCount::Literal("1");
        if let Some((star, digits)) = icon {
            is_star = star;
            // `energyIcons(1)` states its count; `energyIcons()` reads it.
            count = if digits.is_empty() {
                IconCount::Stat(stats.get(&key))
            } else {
                IconCount::Literal(digits)
            };
        }
        let kind = if is_star { "star" } else { "energy" };
        let count_str = count.int_str();
        if markup {
            if let Some(n) = &count_str {
                return format!("[{kind}]{n}[/{kind}]");
            }
        }
        // Plain text has no glyphs, so it spells the resource out — matching the
        // wiki's pre-resolved `description_plain` (extract_cards.py), which is
        // the OTHER text the same card can display. The two must agree or a card
        // reads differently depending on which fallback the backend took.
        if is_pseudo {
            // `energyPrefix` is a bare unit: the number is already in the
            // surrounding text ("a 0{energyPrefix…} copy" -> "a 0 Energy copy"),
            // so its literal count is an icon count, never a value to print.
            return if is_star {
                "★".to_string()
            } else {
                " Energy".to_string()
            };
        }
        let noun = if is_star { "Stars" } else { "Energy" };
        return match count_str {
            Some(n) => format!("{n} {noun}"),
            None => noun.to_string(),
        };
    }

    if let Some(plural_body) = rest.strip_prefix("plural:") {
        let halves = split_top_level(plural_body, '|');
        let singular = halves[0];
        let plural = halves.get(1).copied().unwrap_or("");
        // `isinstance(val, (int, float)) and val == 1` — `True == 1` in Python.
        let is_one = stats
            .get(&key)
            .is_some_and(|v| py::num(v).is_some() && py::eq(v, &Value::from(1)));
        let chosen = if is_one { singular } else { plural };
        return render_template(chosen, stats, flags, markup);
    }

    // Plain value substitution (`{Var}` / `{Var:diff()}`). Unknown vars fall
    // back to their bare name.
    match stats.get(&key) {
        None | Some(Value::Null) => head.to_string(),
        Some(val) => {
            // A float that is integral prints as an int (`6.0` → `6`).
            if let Value::Number(n) = val {
                if !n.is_i64() && !n.is_u64() {
                    if let Some(f) = n.as_f64() {
                        if f.is_finite() && f.fract() == 0.0 {
                            return py::int_str(val).unwrap_or_else(|| py::to_str(val));
                        }
                    }
                }
            }
            py::to_str(val)
        }
    }
}

/// Collect the lowercased `stats` keys `text` would look up (see
/// [`required_stat_vars`]). Mirrors `render_template`'s brace walk and
/// `eval_directive`'s dispatch, following the conditional branch the flags
/// actually select so only rendered vars are reported.
fn collect_vars(text: &str, flags: &Map<String, Value>, out: &mut BTreeSet<String>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        let Some(j) = matching_close(text, i) else {
            break;
        };
        collect_directive(&text[i + 1..j], flags, out);
        i = j + 1;
    }
}

fn collect_directive(body: &str, flags: &Map<String, Value>, out: &mut BTreeSet<String>) {
    let (head, rest) = body.split_once(':').unwrap_or((body, ""));
    let key = py::strip(head).to_lowercase();
    // `choose(...)` picks one branch by a morph value carried in `flags` (not
    // `stats` here — required_stat_vars has no values, only the flags), so
    // follow only the selected branch's vars.
    if let Some(chosen) = choose_branch(rest, flags.get(&key)) {
        // A chooser the caller can't supply is a REQUIRED var, so the card takes
        // the pre-resolved-plain fallback instead of a branch picked by default.
        // `choose_branch` resolves an unknown value to branch 0, which for
        // Shiv's v0.107.1 `{TargetType:choose(AllEnemies): to ALL enemies|}` is
        // the Fan-of-Knives morph: every plain Shiv would have read "Deal 4
        // damage to ALL enemies." Only Mad Science's `cardtype` is supplied.
        if !flags.contains_key(&key) {
            out.insert(key);
        }
        collect_vars(chosen, flags, out);
        return;
    }
    if let Some(flag) = flags.get(&key) {
        let branches = rest.strip_prefix("show:").unwrap_or(rest);
        let halves = split_top_level(branches, '|');
        let chosen = if py::truthy(flag) {
            halves[0]
        } else {
            halves.get(1).copied().unwrap_or("")
        };
        collect_vars(chosen, flags, out);
        return;
    }
    if let Some(plural_body) = rest.strip_prefix("plural:") {
        if !PSEUDO_VARS.contains(&key.as_str()) {
            out.insert(key);
        }
        for half in split_top_level(plural_body, '|') {
            collect_vars(half, flags, out);
        }
        return;
    }
    let restl = rest.to_lowercase();
    if PSEUDO_VARS.contains(&key.as_str())
        || restl.contains("energyicons")
        || restl.contains("staricons")
    {
        return;
    }
    out.insert(key);
}

/// The keyword arguments of [`resolve_card_text`] / [`required_stat_vars`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CardTextFlags<'a> {
    /// Resolve `{InCombat:..}` to its then-branch (else otherwise). False for
    /// shop / deck / reward displays.
    pub in_combat: bool,
    /// Resolve `{IfUpgraded:..}` to its then-branch.
    pub upgraded: bool,
    /// Per-card morph variables (an object) — booleans treated as `{Cond:..}`
    /// conditions plus the `choose` selector value (e.g.
    /// `{"cardtype": "Skill", "hasrider": true, "sapping": true}`).
    pub extra_flags: Option<&'a Value>,
    /// Leave the BBCode `[tag]` markup in place instead of stripping it. The
    /// game's own localization marks its keywords up as
    /// `[gold]Vulnerable[/gold]`, and the card skin renders them gold — so the
    /// card frame asks for the marked-up form while every other caller (action
    /// labels, logs, tooltips) keeps the plain one.
    pub keep_markup: bool,
}

impl CardTextFlags<'_> {
    /// `{"incombat": in_combat, "ifupgraded": upgraded, **extra_flags}`.
    fn to_map(self) -> Map<String, Value> {
        let mut flags = Map::new();
        flags.insert("incombat".to_string(), Value::Bool(self.in_combat));
        flags.insert("ifupgraded".to_string(), Value::Bool(self.upgraded));
        if let Some(Value::Object(extra)) = self.extra_flags {
            for (k, v) in extra {
                flags.insert(k.clone(), v.clone());
            }
        }
        flags
    }
}

/// Return the lowercased `stats` keys `description` needs to fully resolve.
///
/// Walks the same `{...}` DSL as [`resolve_card_text`], following the
/// `InCombat` / `IfUpgraded` branch the flags select, and excluding icon /
/// prefix pseudo-tokens (`energyPrefix`, `energyIcons`, `starIcons`) that
/// don't come from `stats`. Callers compare the result against the `stats` a
/// backend supplied: if some var is missing, the backend couldn't fully
/// resolve the card and the pre-resolved plain description should be used
/// instead (the Rust backend can't attach a card's full `stats` in combat).
#[must_use]
pub fn required_stat_vars(description: Option<&str>, flags: CardTextFlags<'_>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Some(description) = description.filter(|d| !d.is_empty()) else {
        return out;
    };
    collect_vars(description, &flags.to_map(), &mut out);
    out
}

/// Strip BBCode colour tags: `\[/?[a-zA-Z0-9_]+\]` (`[gold] ... [/gold]`).
fn strip_tags(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let mut j = i + 1;
            if j < bytes.len() && bytes[j] == b'/' {
                j += 1;
            }
            let name_start = j;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            if j > name_start && j < bytes.len() && bytes[j] == b']' {
                i = j + 1;
                continue;
            }
        }
        // Not a tag: copy one char (multi-byte safe).
        let ch_len = text[i..].chars().next().map_or(1, char::len_utf8);
        out.push_str(&text[i..i + ch_len]);
        i += ch_len;
    }
    out
}

/// `re.sub(r"\s{2,}", " ", text)`.
fn collapse_multispace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = String::new();
    for ch in text.chars() {
        if py::is_space(ch) {
            run.push(ch);
        } else {
            if run.chars().count() >= 2 {
                out.push(' ');
            } else {
                out.push_str(&run);
            }
            run.clear();
            out.push(ch);
        }
    }
    if run.chars().count() >= 2 {
        out.push(' ');
    } else {
        out.push_str(&run);
    }
    out
}

/// Resolve a card description template to plain display text.
///
/// Resolves the STS2 template DSL: `{Var:diff()}` value substitution from
/// `stats` (keyed by lowercased var name), `{Var:plural:singular|plural}`
/// selection, the `{InCombat:then|else}` / `{IfUpgraded:show:then|else}`
/// boolean wrappers, and the morph directives `{Var:choose(a|b|c):...}`
/// (branch by a value in `stats`/`extra_flags`) + `{Cond:then|else}` for any
/// boolean in `extra_flags` — used by Mad Science's CardType + riders.
/// Branches may nest. BBCode `[tag]` markup is stripped; the energy-icon
/// prefix pseudo-token renders empty. Unknown variables fall back to their
/// name.
///
/// * `description`: raw template, e.g. `"Deal {Damage:diff()} damage."`.
/// * `stats`: resolved values as an object, e.g. `{"damage": 6}`. May be
///   `None` (or any non-object, which reads as empty).
///
/// Returns plain text, e.g. `"Deal 6 damage."`. Empty string for empty input.
#[must_use]
pub fn resolve_card_text(
    description: Option<&str>,
    stats: Option<&Value>,
    flags: CardTextFlags<'_>,
) -> String {
    let Some(description) = description.filter(|d| !d.is_empty()) else {
        return String::new();
    };
    let empty = Map::new();
    let stats = match stats {
        Some(Value::Object(m)) => m,
        _ => &empty,
    };
    let out = render_template(description, stats, &flags.to_map(), flags.keep_markup);
    let out = if flags.keep_markup {
        out
    } else {
        strip_tags(&out)
    };
    py::strip(&collapse_multispace(&out)).to_string()
}

// ── Action labels ────────────────────────────────────────────────────────────

/// Return the item at stable `index` (or positional) `idx`, or `None`.
///
/// Positional fallback applies to integer `idx` only: Python would raise on a
/// float or string subscript, which has no equivalent here.
fn find<'a>(items: Option<&'a Value>, idx: Option<&Value>) -> Option<&'a Value> {
    let items = py::list(items);
    let idx = idx.filter(|v| !v.is_null())?;
    if items.is_empty() {
        return None;
    }
    for it in items {
        if it.is_object() && py::get_raw(it, "index").is_some_and(|v| py::eq(v, idx)) {
            return Some(it);
        }
    }
    let i = py::as_int(idx)?;
    usize::try_from(i).ok().and_then(|i| items.get(i))
}

/// Return the name of the item at stable `index` (or positional) `idx`.
fn named(items: Option<&Value>, idx: Option<&Value>, kind: &str) -> String {
    let fallback = || format!("{kind} {}", py::opt_str(idx));
    match find(items, idx) {
        Some(it) => py::get_raw(it, "name").map_or_else(fallback, py::to_str),
        None => fallback(),
    }
}

/// [`named`] for an enemy target: the enemy's label among `items` — its name
/// plus `" #k"` when several enemies share it ([`crate::enemy_labels`]), so
/// "Play Strike → Two-Tailed Rat #2" says WHICH rat.
///
/// Unlike [`find`], the positional fallback applies only to a list with no
/// stable `index` at all: when the view carries indices, a missing one is a
/// slot that is not on the board, and its list position names a DIFFERENT
/// enemy.
fn enemy_named(items: Option<&Value>, idx: Option<&Value>) -> String {
    let fallback = || format!("enemy {}", py::opt_str(idx));
    let list = py::list(items);
    let Some(want) = idx.filter(|v| !v.is_null()) else {
        return fallback();
    };
    let pos = list
        .iter()
        .position(|e| e.is_object() && py::get_raw(e, "index").is_some_and(|v| py::eq(v, want)))
        .or_else(|| {
            let indexed = list
                .iter()
                .any(|e| e.is_object() && py::get_raw(e, "index").is_some());
            let i = py::as_int(want)?;
            usize::try_from(i)
                .ok()
                .filter(|&i| !indexed && i < list.len())
        });
    match pos {
        Some(pos) if py::get_raw(&list[pos], "name").is_some() => {
            crate::enemy_labels::enemy_labels(list).swap_remove(pos)
        }
        _ => fallback(),
    }
}

/// Like [`named`] but marks upgraded cards with a trailing `+`.
fn card_name(cards: Option<&Value>, idx: Option<&Value>) -> String {
    let Some(it) = find(cards, idx) else {
        return format!("card {}", py::opt_str(idx));
    };
    let name =
        py::get_raw(it, "name").map_or_else(|| format!("card {}", py::opt_str(idx)), py::to_str);
    if py::truthy_opt(py::get_raw(it, "upgraded")) {
        format!("{name}+")
    } else {
        name
    }
}

/// `next((args[k] for k in keys if args.get(k) is not None), None)`.
fn first_present<'a>(args: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .find_map(|k| args.get(*k).filter(|v| !v.is_null()))
}

/// Render the action taken from `state` as a short English phrase.
///
/// * `state`: the decision-point state the action was chosen from (used to
///   resolve card indices to names).
/// * `action`: the logged action `data` dict (`{cmd, action, args}`).
///
/// Returns e.g. `"Play Strike → enemy 0"`, `"End turn"`, `"Start run ·
/// Ironclad"`. Empty string when there is no action.
#[must_use]
pub fn describe_action(state: Option<&Value>, action: Option<&Value>) -> String {
    let Some(action) = action.filter(|a| py::truthy(a)) else {
        return String::new();
    };
    let null = Value::Null;
    let state = match state {
        Some(s) if py::truthy(s) && s.is_object() => s,
        _ => &null,
    };
    // A pre-rendered label wins (live-game human actions carry only a display
    // string -- realgame_workbench_log.log_human_action).
    if let Some(label) = py::get(action, "label").filter(|l| py::truthy(l)) {
        return py::to_str(label);
    }
    let cmd = py::get(action, "cmd");
    if cmd.and_then(Value::as_str) == Some("start_run") {
        let character =
            py::get_raw(action, "character").map_or_else(|| "?".to_string(), py::to_str);
        return format!("Start run · {character}");
    }

    let mut act: Option<Value> = match py::get_raw(action, "action") {
        Some(Value::Null) => None,
        Some(v) => Some(v.clone()),
        None => cmd.cloned(),
    };
    let mut args: Map<String, Value> = match py::get(action, "args") {
        Some(Value::Object(m)) if !m.is_empty() => m.clone(),
        _ => Map::new(),
    };
    if act.as_ref().and_then(Value::as_str) == Some(NATIVE_ACTION) {
        // A native-agent step recorded before the writer persisted `label`
        // (workbench runs up to 2026-08-26). Its real kind sits inside `args`;
        // left wrapped, every autoplayed combat step in the replay's history
        // read "__agent_native__" where the live log had said "Play Strike →
        // Twig Slime". Newer logs never reach here — their `label` wins above.
        let kind = args
            .get("action")
            .filter(|v| py::truthy(v))
            .map_or_else(String::new, py::to_str);
        let verb = NATIVE_KIND_VERB
            .iter()
            .find(|(k, _)| *k == kind)
            .map_or_else(
                || {
                    if kind.is_empty() {
                        "?".to_string()
                    } else {
                        kind.clone()
                    }
                },
                |(_, v)| (*v).to_string(),
            );
        act = Some(Value::String(verb));
        args.remove("action");
    }
    let act_str = act.as_ref().and_then(Value::as_str);
    match act_str {
        Some("play_card") => {
            // Both index spellings: the C# verb's `card_index` and the raw
            // RunAction's `hand_idx` (same semantics — a position in the hand).
            let card_idx = first_present(&args, &["card_index", "hand_idx"]);
            let name = card_name(py::get(state, "hand"), card_idx);
            match args.get("target_index").filter(|v| !v.is_null()) {
                Some(target) => {
                    format!(
                        "Play {name} → {}",
                        enemy_named(py::get(state, "enemies"), Some(target))
                    )
                }
                None => format!("Play {name}"),
            }
        }
        Some("end_turn") => "End turn".to_string(),
        Some("use_potion" | "drink_potion") => {
            // Both index spellings: `to_workbench_verb` emits `potion_index`,
            // the native RunAction shape `slot`.
            let slot = first_present(&args, &["potion_index", "slot"]);
            // The BELT, not `state["potions"]` — that key is the shop's stock
            // (see `buy_potion` below), so reading it here named a drink after
            // whatever happened to be on sale. Top level is the fallback for a
            // backend that puts the belt there.
            let belt = py::get(py::get_obj(state, "player"), "potions")
                .filter(|v| py::truthy(v))
                .or_else(|| py::get(state, "potions"));
            let name = named(belt, slot, "potion");
            match args.get("target_index").filter(|v| !v.is_null()) {
                Some(target) => {
                    format!(
                        "Drink {name} → {}",
                        enemy_named(py::get(state, "enemies"), Some(target))
                    )
                }
                None => format!("Drink {name}"),
            }
        }
        Some("select_map_node") => format!(
            "Map → ({},{})",
            py::opt_str(args.get("col")),
            py::opt_str(args.get("row"))
        ),
        Some("select_card_reward") => {
            format!(
                "Take {}",
                card_name(py::get(state, "cards"), args.get("card_index"))
            )
        }
        Some("skip_card_reward") => "Skip card reward".to_string(),
        Some("buy_card") => {
            format!(
                "Buy {}",
                card_name(py::get(state, "cards"), args.get("card_index"))
            )
        }
        Some(act @ ("buy_relic" | "buy_potion")) => {
            let (key, idx_arg) = if act == "buy_relic" {
                ("relics", "relic_index")
            } else {
                ("potions", "potion_index")
            };
            format!(
                "Buy {}",
                named(
                    py::get(state, key),
                    args.get(idx_arg),
                    &key[..key.len() - 1]
                )
            )
        }
        Some("choose_option") => {
            let oi = args.get("option_index");
            let options = py::get_list(state, "options");
            if let Some(opt) = oi
                .and_then(py::as_int)
                .and_then(|i| usize::try_from(i).ok())
                .and_then(|i| options.get(i))
            {
                // C# options carry `text`/`name`; the Rust backend emits `title`
                // (generic screens) or `option_id` (rest sites).
                let label = if opt.is_object() {
                    ["text", "name", "title", "option_id"]
                        .iter()
                        .find_map(|k| py::get(opt, k).filter(|v| py::truthy(v)))
                        .map(py::to_str)
                } else {
                    Some(py::to_str(opt)).filter(|s| !s.is_empty())
                };
                if let Some(label) = label {
                    return format!("Choose: {label}");
                }
            }
            format!("Choose option {}", py::opt_str(oi))
        }
        Some("select_cards") => {
            let indices = args.get("indices").map_or_else(String::new, py::to_str);
            let idxs: Vec<Value> = indices
                .split(',')
                .filter(|t| {
                    let t = py::strip(t);
                    !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit())
                })
                .filter_map(|t| py::strip(t).parse::<i64>().ok())
                .map(Value::from)
                .collect();
            // A SHOP's only `select_cards` is the removal service, and its
            // indices are MASTER-DECK positions (`run_policy.to_workbench_verb`
            // maps `shop_remove_card{deck_idx}` here) — while a shop state's
            // `cards` is the STOCK. Naming them against the stock labelled a
            // removal with whatever happened to be on sale at that index: report
            // wb-c2adad9c8eeb21f7 logged "Select Thunderclap" for a 75g removal
            // of deck card #0. Say which deck card, the way the PPO panel
            // already does (`ppo_policy.shop_remove_label`).
            let deck = py::get(py::get_obj(state, "player"), "deck").filter(|d| py::truthy(d));
            if py::get(state, "decision").and_then(Value::as_str) == Some("shop") {
                if let Some(deck) = deck {
                    let named: Vec<String> = idxs
                        .iter()
                        .map(|i| {
                            format!(
                                "{} (deck #{})",
                                card_name(Some(deck), Some(i)),
                                py::to_str(i)
                            )
                        })
                        .collect();
                    return if named.is_empty() {
                        "Remove card".to_string()
                    } else {
                        format!("Remove {}", named.join(", "))
                    };
                }
            }
            let names: Vec<String> = idxs
                .iter()
                .map(|i| card_name(py::get(state, "cards"), Some(i)))
                .collect();
            // The Rust backend stamps the card_select's purpose on `context`
            // ("upgrade" = rest-site Smith, "remove" = deck removal, otherwise a
            // combat pending-choice kind).
            let context = py::get(state, "context")
                .filter(|v| py::truthy(v))
                .map_or_else(String::new, py::to_str);
            let verb = match context.as_str() {
                "upgrade" => "Upgrade",
                "remove" => "Remove",
                _ => "Select",
            };
            if names.is_empty() {
                "Select cards".to_string()
            } else {
                format!("{verb} {}", names.join(", "))
            }
        }
        Some("skip_select") => "Skip selection".to_string(),
        _ => match act {
            Some(v) if py::truthy(&v) => py::to_str(&v),
            _ => "?".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    //! Port of `tests/workbench/test_render.py` (every case, same names).

    use super::*;
    use serde_json::json;

    fn resolve(tmpl: &str, stats: Value) -> String {
        resolve_card_text(Some(tmpl), Some(&stats), CardTextFlags::default())
    }

    fn resolve_with(tmpl: &str, stats: Value, flags: CardTextFlags<'_>) -> String {
        resolve_card_text(Some(tmpl), Some(&stats), flags)
    }

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn test_resolve_simple_damage() {
        assert_eq!(
            resolve("Deal {Damage:diff()} damage.", json!({"damage": 6})),
            "Deal 6 damage."
        );
    }

    #[test]
    fn test_resolve_strips_bbcode() {
        assert_eq!(
            resolve(
                "Gain {Block:diff()} [gold]Block[/gold].",
                json!({"block": 5})
            ),
            "Gain 5 Block."
        );
    }

    #[test]
    fn test_resolve_keeps_bbcode_when_asked() {
        // The card skin colours the game's own keyword markup, so it needs the
        // tags left in — while every other caller keeps the stripped form above.
        let flags = CardTextFlags {
            keep_markup: true,
            ..Default::default()
        };
        assert_eq!(
            resolve_with(
                "Gain {Block:diff()} [gold]Block[/gold].",
                json!({"block": 5}),
                flags
            ),
            "Gain 5 [gold]Block[/gold]."
        );
    }

    #[test]
    fn test_resolve_cost_glyphs_in_markup_mode() {
        // All four icon-token shapes the game's card text uses. The count is
        // either read from `stats` (empty parens) or stated literally in the
        // formatter; `singleStarIcon` carries no formatter at all. Getting the
        // literal forms wrong is invisible in plain text — they render empty —
        // which is why they are pinned here.
        let flags = CardTextFlags {
            keep_markup: true,
            ..Default::default()
        };
        assert_eq!(
            resolve_with("Gain {Energy:energyIcons()}.", json!({"energy": 3}), flags),
            "Gain [energy]3[/energy]."
        );
        assert_eq!(
            resolve_with(
                "Add a 0{energyPrefix:energyIcons(1)} copy.",
                json!({}),
                flags
            ),
            "Add a 0[energy]1[/energy] copy."
        );
        assert_eq!(
            resolve_with("Gain {Stars:starIcons()}.", json!({"stars": 2}), flags),
            "Gain [star]2[/star]."
        );
        assert_eq!(
            resolve_with("Spend {singleStarIcon} to win.", json!({}), flags),
            "Spend [star]1[/star] to win."
        );
    }

    #[test]
    fn test_plain_mode_spells_the_cost_resource_out() {
        // Plain text has no glyphs, so it names the resource. These strings are
        // matched against the wiki's pre-resolved `description_plain`
        // (`extract_cards.py`), which is the OTHER text the same card can
        // display — the backend falls back to it whenever a card's `stats` are
        // incomplete. If the two disagree, a card silently reads differently
        // depending on which path served it.
        assert_eq!(
            resolve("Gain {Energy:energyIcons()}.", json!({"energy": 1})),
            "Gain 1 Energy."
        );
        assert_eq!(
            resolve("Gain {Stars:starIcons()}.", json!({"stars": 2})),
            "Gain 2 Stars."
        );
        // energyPrefix is a bare unit — the number is already in the sentence.
        assert_eq!(
            resolve("Add a 0{energyPrefix:energyIcons(1)} copy.", json!({})),
            "Add a 0 Energy copy."
        );
        // Previously rendered as nothing, leaving "Whenever you spend , ...".
        assert_eq!(
            resolve(
                "Whenever you spend {singleStarIcon}, gain 3 Block.",
                json!({})
            ),
            "Whenever you spend ★, gain 3 Block."
        );
        // An unresolved count still names the resource rather than leaking the var.
        assert_eq!(
            resolve("Gain {Energy:energyIcons()}.", json!({})),
            "Gain Energy."
        );
    }

    #[test]
    fn test_resolve_plural_singular_and_plural() {
        let tmpl = "Draw {Cards} {Cards:plural:card|cards}.";
        assert_eq!(resolve(tmpl, json!({"cards": 3})), "Draw 3 cards.");
        assert_eq!(resolve(tmpl, json!({"cards": 1})), "Draw 1 card.");
    }

    #[test]
    fn test_resolve_unknown_var_falls_back_to_name() {
        assert_eq!(
            resolve("Deal {Damage} damage.", json!({})),
            "Deal Damage damage."
        );
    }

    #[test]
    fn test_resolve_empty() {
        assert_eq!(
            resolve_card_text(Some(""), None, CardTextFlags::default()),
            ""
        );
        assert_eq!(resolve_card_text(None, None, CardTextFlags::default()), "");
    }

    #[test]
    fn test_resolve_incombat_conditional() {
        // Body Slam: the combat-only parenthetical lives in {InCombat:then|else};
        // out of combat (shop/deck) takes the empty else-branch.
        let tmpl = "Deal damage equal to your [gold]Block[/gold].{InCombat:\n(Deals {CalculatedDamage:diff()} damage)|}";
        assert_eq!(
            resolve_with(
                tmpl,
                json!({}),
                CardTextFlags {
                    in_combat: false,
                    ..Default::default()
                }
            ),
            "Deal damage equal to your Block."
        );
        assert_eq!(
            resolve_with(
                tmpl,
                json!({"calculateddamage": 18}),
                CardTextFlags {
                    in_combat: true,
                    ..Default::default()
                }
            ),
            "Deal damage equal to your Block.\n(Deals 18 damage)"
        );
    }

    #[test]
    fn test_resolve_ifupgraded_show_branch() {
        let tmpl = "[gold]Upgrade[/gold] {IfUpgraded:show:ALL cards|a card} in your Hand.";
        assert_eq!(
            resolve_with(
                tmpl,
                json!({}),
                CardTextFlags {
                    upgraded: false,
                    ..Default::default()
                }
            ),
            "Upgrade a card in your Hand."
        );
        assert_eq!(
            resolve_with(
                tmpl,
                json!({}),
                CardTextFlags {
                    upgraded: true,
                    ..Default::default()
                }
            ),
            "Upgrade ALL cards in your Hand."
        );
    }

    #[test]
    fn test_resolve_nested_plural_and_conditional() {
        // CHARGE: a {Cards:plural:..|..} whose branches each nest an
        // {IfUpgraded:..} — exercises balanced-brace recursion (a naive split
        // would mangle it).
        let tmpl = "Transform into {Cards:plural:{IfUpgraded:show:Bomb+|Bomb}|{IfUpgraded:show:Bombs+|Bombs}}.";
        assert_eq!(
            resolve_with(
                tmpl,
                json!({"cards": 1}),
                CardTextFlags {
                    upgraded: false,
                    ..Default::default()
                }
            ),
            "Transform into Bomb."
        );
        assert_eq!(
            resolve_with(
                tmpl,
                json!({"cards": 2}),
                CardTextFlags {
                    upgraded: true,
                    ..Default::default()
                }
            ),
            "Transform into Bombs+."
        );
    }

    // ── Mad Science morph DSL ({CardType:choose(...)} + rider booleans) ──────

    /// The real MAD_SCIENCE.description template, used across the morph tests.
    const MAD_SCIENCE: &str = concat!(
        "{CardType:choose(Attack|Skill|Power):Deal {Damage:diff()} damage",
        "{Violence: {ViolenceHits:diff()} times|}.|Gain {Block:diff()} [gold]Block[/gold].|}",
        "{HasRider:{Sapping:\nApply {SappingWeak:diff()} [gold]Weak[/gold].\n",
        "Apply {SappingVulnerable:diff()} [gold]Vulnerable[/gold].|}",
        "{Choking:\nWhenever you play a card this turn, the enemy loses {ChokingDamage:diff()} HP.|}",
        "{Energized:\nGain {EnergizedEnergy:energyIcons()}.|}",
        "{Wisdom:\nDraw {WisdomCards:diff()} cards.|}",
        "{Chaos:\nAdd a random card into your [gold]Hand[/gold]. ",
        "It costs 0 {energyPrefix:energyIcons(1)} this turn.|}",
        "{Expertise:Gain {ExpertiseStrength:diff()} [gold]Strength[/gold].\n",
        "Gain {ExpertiseDexterity:diff()} [gold]Dexterity[/gold].|}",
        "{Curious:Powers cost {CuriousReduction:diff()} {energyPrefix:energyIcons(1)} less.|}",
        "{Improvement:At the end of combat, [gold]Upgrade[/gold] a random card.|}",
        "|{CardType:choose(Attack|Skill|Power):\n???|\n???|???}}"
    );

    /// Build the extra_flags dict the way the server does from the C# fields.
    fn morph(card_type: &str, rider: &str) -> Value {
        let riders = [
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
        let mut flags = Map::new();
        for r in riders {
            flags.insert(r.to_string(), Value::Bool(rider.to_lowercase() == r));
        }
        flags.insert(
            "hasrider".to_string(),
            Value::Bool(rider.to_lowercase() != "none"),
        );
        flags.insert("cardtype".to_string(), Value::String(card_type.to_string()));
        Value::Object(flags)
    }

    fn with_extra(extra: &Value) -> CardTextFlags<'_> {
        CardTextFlags {
            extra_flags: Some(extra),
            ..Default::default()
        }
    }

    #[test]
    fn test_choose_selects_branch_by_value() {
        let tmpl = "{CardType:choose(Attack|Skill|Power):attack text|skill text|power text}";
        assert_eq!(resolve(tmpl, json!({"cardtype": "Attack"})), "attack text");
        assert_eq!(resolve(tmpl, json!({"cardtype": "Skill"})), "skill text");
        assert_eq!(resolve(tmpl, json!({"cardtype": "Power"})), "power text");
    }

    #[test]
    fn test_choose_defaults_to_first_branch_when_value_missing() {
        let tmpl = "{CardType:choose(Attack|Skill|Power):a|b|c}";
        assert_eq!(resolve(tmpl, json!({})), "a");
    }

    #[test]
    fn test_generic_boolean_conditional_via_extra_flags() {
        let tmpl = "Deal 6 damage{Violence: {ViolenceHits:diff()} times|}.";
        let on = json!({"violence": true});
        let off = json!({"violence": false});
        assert_eq!(
            resolve_with(tmpl, json!({"violencehits": 3}), with_extra(&on)),
            "Deal 6 damage 3 times."
        );
        assert_eq!(
            resolve_with(tmpl, json!({"violencehits": 3}), with_extra(&off)),
            "Deal 6 damage."
        );
    }

    #[test]
    fn test_mad_science_attack_sapping() {
        let stats =
            json!({"cardtype": "Attack", "damage": 12, "sappingweak": 2, "sappingvulnerable": 2});
        let extra = morph("Attack", "Sapping");
        let out = resolve_with(MAD_SCIENCE, stats, with_extra(&extra));
        assert_eq!(out, "Deal 12 damage.\nApply 2 Weak.\nApply 2 Vulnerable.");
    }

    #[test]
    fn test_mad_science_attack_violence_rider() {
        let stats = json!({"cardtype": "Attack", "damage": 12, "violencehits": 3});
        let extra = morph("Attack", "Violence");
        let out = resolve_with(MAD_SCIENCE, stats, with_extra(&extra));
        assert_eq!(out, "Deal 12 damage 3 times.");
    }

    #[test]
    fn test_mad_science_skill_no_rider_shows_mystery() {
        // No rider → HasRider false → the else-branch shows the "???"
        // placeholder (the game's pre-reveal preview), chosen by card type.
        let stats = json!({"cardtype": "Skill", "block": 8});
        let extra = morph("Skill", "None");
        let out = resolve_with(MAD_SCIENCE, stats, with_extra(&extra));
        assert_eq!(out, "Gain 8 Block.\n???");
    }

    #[test]
    fn test_mad_science_power_expertise() {
        let stats = json!({"cardtype": "Power", "expertisestrength": 2, "expertisedexterity": 2});
        let extra = morph("Power", "Expertise");
        let out = resolve_with(MAD_SCIENCE, stats, with_extra(&extra));
        assert_eq!(out, "Gain 2 Strength.\nGain 2 Dexterity.");
    }

    #[test]
    fn test_required_stat_vars_follows_choose_and_rider() {
        // Skill+no-rider needs block; the rider/attack vars are off-branch.
        let extra = morph("Skill", "None");
        let need = required_stat_vars(Some(MAD_SCIENCE), with_extra(&extra));
        assert_eq!(need, set(&["block"]));
        // Attack+Sapping needs damage + the two sapping vars, not block.
        let extra2 = morph("Attack", "Sapping");
        let need2 = required_stat_vars(Some(MAD_SCIENCE), with_extra(&extra2));
        assert_eq!(need2, set(&["damage", "sappingweak", "sappingvulnerable"]));
    }

    #[test]
    fn test_describe_play_card_uses_hand_name_and_target() {
        let state = json!({"hand": [{"index": 0, "name": "Strike"}]});
        let action = json!({"cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 1}});
        assert_eq!(
            describe_action(Some(&state), Some(&action)),
            "Play Strike → enemy 1"
        );
    }

    #[test]
    fn test_describe_target_numbers_a_duplicated_enemy_name() {
        // Slot 2 died, so the view lists slots 0, 1, 3, 4: rats #1..#3 + Louse.
        let state = json!({
            "hand": [{"index": 0, "name": "Strike"}],
            "player": {"potions": [{"index": 0, "name": "Fire Potion"}]},
            "enemies": [
                {"index": 0, "name": "Two-Tailed Rat"},
                {"index": 1, "name": "Louse"},
                {"index": 3, "name": "Two-Tailed Rat"},
                {"index": 4, "name": "Two-Tailed Rat"},
            ],
        });
        let play = |t: i64| {
            describe_action(
                Some(&state),
                Some(&json!({"cmd": "action", "action": "play_card",
                             "args": {"card_index": 0, "target_index": t}})),
            )
        };
        assert_eq!(play(0), "Play Strike → Two-Tailed Rat #1");
        assert_eq!(play(1), "Play Strike → Louse");
        assert_eq!(play(3), "Play Strike → Two-Tailed Rat #2");
        assert_eq!(play(4), "Play Strike → Two-Tailed Rat #3");
        let drink = json!({"cmd": "action", "action": "use_potion",
                           "args": {"potion_index": 0, "target_index": 4}});
        assert_eq!(
            describe_action(Some(&state), Some(&drink)),
            "Drink Fire Potion → Two-Tailed Rat #3"
        );
        // Slot 2 is not on the board: no label, not the slot-3 rat at list
        // position 2.
        assert_eq!(play(2), "Play Strike → enemy 2");
        // Without stable indices the list position still resolves.
        let bare = json!({
            "hand": [{"index": 0, "name": "Strike"}],
            "enemies": [{"name": "Two-Tailed Rat"}, {"name": "Two-Tailed Rat"}],
        });
        let act = json!({"cmd": "action", "action": "play_card",
                         "args": {"card_index": 0, "target_index": 1}});
        assert_eq!(
            describe_action(Some(&bare), Some(&act)),
            "Play Strike → Two-Tailed Rat #2"
        );
    }

    #[test]
    fn test_describe_end_turn_and_start_run() {
        let empty = json!({});
        let end = json!({"cmd": "action", "action": "end_turn"});
        assert_eq!(describe_action(Some(&empty), Some(&end)), "End turn");
        let start = json!({"cmd": "start_run", "character": "Ironclad"});
        assert_eq!(
            describe_action(Some(&empty), Some(&start)),
            "Start run · Ironclad"
        );
    }

    #[test]
    fn test_describe_select_card_reward() {
        let state = json!({"cards": [{"index": 0, "name": "Anger"}]});
        let action =
            json!({"cmd": "action", "action": "select_card_reward", "args": {"card_index": 0}});
        assert_eq!(describe_action(Some(&state), Some(&action)), "Take Anger");
    }

    #[test]
    fn test_describe_play_upgraded_card_appends_plus() {
        // Upgraded cards are flagged with `upgraded: True`; the log marks them
        // with a trailing "+" (e.g. an upgraded Bash reads "Bash+").
        let state = json!({"hand": [{"index": 0, "name": "Bash", "upgraded": true}]});
        let action = json!({"cmd": "action", "action": "play_card", "args": {"card_index": 0}});
        assert_eq!(describe_action(Some(&state), Some(&action)), "Play Bash+");
    }

    #[test]
    fn test_describe_play_unupgraded_card_no_plus() {
        let state = json!({"hand": [{"index": 0, "name": "Bash", "upgraded": false}]});
        let action = json!({"cmd": "action", "action": "play_card", "args": {"card_index": 0}});
        assert_eq!(describe_action(Some(&state), Some(&action)), "Play Bash");
    }

    #[test]
    fn test_describe_take_upgraded_card_appends_plus() {
        let state = json!({"cards": [{"index": 0, "name": "Anger", "upgraded": true}]});
        let action =
            json!({"cmd": "action", "action": "select_card_reward", "args": {"card_index": 0}});
        assert_eq!(describe_action(Some(&state), Some(&action)), "Take Anger+");
    }

    #[test]
    fn test_describe_choose_option_uses_title() {
        // The Rust backend emits event options with a `title` field (C# uses
        // `text`/`name`); describe_action should read it so event picks read
        // "Choose: <option>" rather than "Choose option 0".
        let state = json!({"options": [{"index": 0, "title": "Lost Wisp"}]});
        let action =
            json!({"cmd": "action", "action": "choose_option", "args": {"option_index": 0}});
        assert_eq!(
            describe_action(Some(&state), Some(&action)),
            "Choose: Lost Wisp"
        );
    }

    #[test]
    fn test_describe_buy_names_the_item() {
        // Shop purchases should log the item name, not the bare verb.
        let shop = json!({
            "cards": [{"index": 0, "name": "Strike"}],
            "relics": [{"index": 0, "name": "Akabeko"}],
            "potions": [{"index": 0, "name": "Fire Potion"}],
        });
        let a = json!({"cmd": "a", "action": "buy_card", "args": {"card_index": 0}});
        assert_eq!(describe_action(Some(&shop), Some(&a)), "Buy Strike");
        let b = json!({"cmd": "a", "action": "buy_relic", "args": {"relic_index": 0}});
        assert_eq!(describe_action(Some(&shop), Some(&b)), "Buy Akabeko");
        let c = json!({"cmd": "a", "action": "buy_potion", "args": {"potion_index": 0}});
        assert_eq!(describe_action(Some(&shop), Some(&c)), "Buy Fire Potion");
    }

    #[test]
    fn test_describe_no_action() {
        let empty = json!({});
        assert_eq!(describe_action(Some(&empty), None), "");
    }

    #[test]
    fn test_required_stat_vars_simple() {
        assert_eq!(
            required_stat_vars(
                Some("Deal {Damage:diff()} damage."),
                CardTextFlags::default()
            ),
            set(&["damage"])
        );
    }

    #[test]
    fn test_required_stat_vars_multiple() {
        // Bash-style: damage plus an applied-power var.
        let tpl = "Deal {Damage:diff()} damage.\nApply {VulnerablePower:diff()} Vulnerable.";
        assert_eq!(
            required_stat_vars(Some(tpl), CardTextFlags::default()),
            set(&["damage", "vulnerablepower"])
        );
    }

    #[test]
    fn test_required_stat_vars_includes_plural_count() {
        let tpl = "Add {Cards:diff()} {Cards:plural:card|cards}.";
        assert_eq!(
            required_stat_vars(Some(tpl), CardTextFlags::default()),
            set(&["cards"])
        );
    }

    #[test]
    fn test_required_stat_vars_excludes_icon_pseudo_tokens() {
        // `energyPrefix` / `energyIcons` aren't stats — only Damage is required.
        let tpl = "Deal {Damage:diff()} damage.\nAdd a 0{energyPrefix:energyIcons(1)} copy.";
        assert_eq!(
            required_stat_vars(Some(tpl), CardTextFlags::default()),
            set(&["damage"])
        );
    }

    #[test]
    fn test_required_stat_vars_follows_ifupgraded_branch() {
        let tpl = "{IfUpgraded:show:Gain {Block:diff()} Block|Deal {Damage:diff()} damage}";
        assert_eq!(
            required_stat_vars(
                Some(tpl),
                CardTextFlags {
                    upgraded: true,
                    ..Default::default()
                }
            ),
            set(&["block"])
        );
        assert_eq!(
            required_stat_vars(
                Some(tpl),
                CardTextFlags {
                    upgraded: false,
                    ..Default::default()
                }
            ),
            set(&["damage"])
        );
    }

    #[test]
    fn test_required_stat_vars_none() {
        assert_eq!(
            required_stat_vars(None, CardTextFlags::default()),
            BTreeSet::new()
        );
        assert_eq!(
            required_stat_vars(Some("Exhaust."), CardTextFlags::default()),
            BTreeSet::new()
        );
    }

    /// Shiv's real v0.107.1 template. Its target type morphs — Fan of Knives
    /// makes every Shiv hit all enemies — and only the live board knows which
    /// it is.
    const SHIV: &str =
        "Deal {Damage:diff()} damage{TargetType:choose(AllEnemies): to ALL enemies|}.";

    #[test]
    fn test_unsuppliable_chooser_is_a_required_var() {
        // A chooser nobody supplies must be REPORTED as required, so the caller
        // takes the pre-resolved-plain fallback. `choose_branch` resolves an
        // unknown value to branch 0. For Shiv that branch is the Fan-of-Knives
        // morph, so a silently-defaulted render says every Shiv hits ALL
        // enemies. Only Mad Science's `cardtype` travels in `extra_flags`;
        // `targettype` does not, and a chooser the renderer cannot answer is
        // exactly the "confidently wrong" case the fallback exists for.
        assert_eq!(
            required_stat_vars(Some(SHIV), CardTextFlags::default()),
            set(&["damage", "targettype"])
        );
        // A supplied chooser stays a morph field, never a required stat.
        let extra = morph("Skill", "None");
        assert!(!required_stat_vars(Some(MAD_SCIENCE), with_extra(&extra)).contains("cardtype"));
    }

    #[test]
    fn test_describe_shop_removal_names_the_deck_card() {
        // A shop card-removal must name the DECK card, not the shop stock.
        // `shop_remove_card{deck_idx}` maps to `select_cards{indices}`
        // (`run_policy.to_workbench_verb`), and a shop state's `cards` is the
        // STOCK — so the generic path labelled a removal with whatever the shop
        // happened to sell at that index. Report wb-c2adad9c8eeb21f7: "play
        // best" removed deck card #0 (gold 143→68, the 75g removal price) and
        // the timeline logged it as "Select Thunderclap", the first card on
        // sale.
        let shop = json!({
            "decision": "shop",
            "cards": [{"index": 0, "name": "Thunderclap"}],
            "player": {"deck": [{"index": 0, "name": "Strike"}, {"index": 1, "name": "Bash"}]},
        });
        let action = json!({"cmd": "action", "action": "select_cards", "args": {"indices": "0"}});
        assert_eq!(
            describe_action(Some(&shop), Some(&action)),
            "Remove Strike (deck #0)"
        );
    }

    #[test]
    fn test_describe_shop_removal_marks_upgraded_and_multiple_indices() {
        let shop = json!({
            "decision": "shop",
            "cards": [{"index": 0, "name": "Thunderclap"}],
            "player": {
                "deck": [
                    {"index": 0, "name": "Strike"},
                    {"index": 1, "name": "Bash", "upgraded": true},
                ]
            },
        });
        let action = json!({"cmd": "action", "action": "select_cards", "args": {"indices": "0,1"}});
        assert_eq!(
            describe_action(Some(&shop), Some(&action)),
            "Remove Strike (deck #0), Bash+ (deck #1)"
        );
    }

    #[test]
    fn test_describe_shop_removal_without_a_deck_keeps_the_generic_label() {
        // A backend that ships no `player.deck` (C#) must not regress to
        // "Remove card 0" — it keeps the pre-existing stock-indexed label.
        let shop = json!({"decision": "shop", "cards": [{"index": 0, "name": "Thunderclap"}]});
        let action = json!({"cmd": "action", "action": "select_cards", "args": {"indices": "0"}});
        assert_eq!(
            describe_action(Some(&shop), Some(&action)),
            "Select Thunderclap"
        );
    }

    #[test]
    fn test_describe_prefers_the_recorded_label() {
        // The workbench's quit path records the label the live log displayed;
        // it wins over anything re-derived from the action shape.
        let action = json!({"cmd": "action", "action": "__agent_native__", "args": {}, "label": "Play Bash+"});
        let empty = json!({});
        assert_eq!(describe_action(Some(&empty), Some(&action)), "Play Bash+");
    }

    #[test]
    fn test_describe_unwraps_a_native_agent_action() {
        // Autoplayed steps recorded before the label existed still read as
        // English. Every native-agent node logs under the `__agent_native__`
        // sentinel with its real kind inside `args`; left wrapped, a whole
        // run's combat history rendered as "__agent_native__".
        let state = json!({"hand": [{"index": 0, "name": "Strike"}], "enemies": [{"index": 1, "name": "Nibbit"}]});
        let native = json!({
            "cmd": "action",
            "action": "__agent_native__",
            "args": {"action": "play", "card_index": 0, "target_index": 1},
        });
        assert_eq!(
            describe_action(Some(&state), Some(&native)),
            "Play Strike → Nibbit"
        );
        let empty = json!({});
        let end = json!({"action": "__agent_native__", "args": {"action": "end_turn"}});
        assert_eq!(describe_action(Some(&empty), Some(&end)), "End turn");
        // A raw RunAction (`play_card`/`hand_idx`) reads too, not just the
        // normalized `play`/`card_index` pair the log writer emits.
        let raw =
            json!({"action": "__agent_native__", "args": {"action": "play_card", "hand_idx": 0}});
        assert_eq!(describe_action(Some(&state), Some(&raw)), "Play Strike");
        // A kind with no dedicated verb degrades to the kind, not to the sentinel.
        let mv = json!({"action": "__agent_native__", "args": {"action": "move", "col": 4}});
        assert_eq!(describe_action(Some(&empty), Some(&mv)), "move");
    }

    #[test]
    fn test_describe_potion() {
        // Both index spellings: the C# verb's `potion_index` and the native
        // RunAction's `slot`. The belt lives under `player.potions`; top-level
        // `potions` is the SHOP's stock, so naming a drink from it labelled it
        // with whatever was on sale.
        let state = json!({
            "player": {"potions": [{"index": 0, "name": "Fire Potion"}]},
            "potions": [{"index": 0, "name": "Attack Potion"}],
            "enemies": [{"index": 0, "name": "Nibbit"}],
        });
        let verb = json!({"cmd": "action", "action": "use_potion", "args": {"potion_index": 0}});
        assert_eq!(
            describe_action(Some(&state), Some(&verb)),
            "Drink Fire Potion"
        );
        let native = json!({
            "action": "__agent_native__",
            "args": {"action": "drink_potion", "slot": 0, "target_index": 0},
        });
        assert_eq!(
            describe_action(Some(&state), Some(&native)),
            "Drink Fire Potion → Nibbit"
        );
    }

    // ── Python-semantics pins (not in the Python suite; guard the helpers) ──

    #[test]
    fn py_float_repr_matches_python() {
        assert_eq!(py::float_repr(2.5), "2.5");
        assert_eq!(py::float_repr(6.0), "6.0");
        assert_eq!(py::float_repr(-6.5), "-6.5");
        assert_eq!(py::float_repr(0.1), "0.1");
        assert_eq!(py::float_repr(0.0001), "0.0001");
        assert_eq!(py::float_repr(0.00001), "1e-05");
        assert_eq!(py::float_repr(1e15), "1000000000000000.0");
        assert_eq!(py::float_repr(1e16), "1e+16");
        assert_eq!(py::float_repr(1.5e16), "1.5e+16");
        assert_eq!(py::float_repr(123.456), "123.456");
        assert_eq!(py::float_repr(0.0), "0.0");
    }

    #[test]
    fn integral_float_stat_prints_as_int_and_bool_as_python_bool() {
        assert_eq!(
            resolve("Deal {Damage} damage.", json!({"damage": 6.0})),
            "Deal 6 damage."
        );
        assert_eq!(
            resolve("Deal {Damage} damage.", json!({"damage": 6.5})),
            "Deal 6.5 damage."
        );
        assert_eq!(resolve("{Flag}", json!({"flag": true})), "True");
    }

    #[test]
    fn multispace_and_tag_stripping_edge_cases() {
        assert_eq!(
            resolve("a  b\t\tc [[gold]x[/gold]]", json!({})),
            "a b c [x]"
        );
        assert_eq!(resolve("[not a tag", json!({})), "[not a tag");
    }

    #[test]
    fn unbalanced_brace_is_emitted_verbatim() {
        assert_eq!(
            resolve("Deal {Damage damage.", json!({"damage": 6})),
            "Deal {Damage damage."
        );
    }
}
