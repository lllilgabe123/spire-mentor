//! `colocate-card-data <char>`: move each card's CardSpec factory + upgrade
//! delta into its per-card module file, regenerate the central reference
//! tables, and delete `<char>_specs_generated.rs`.

use quote::ToTokens;
use syn::spanned::Spanned;

#[derive(Debug)]
pub struct SpecFactory {
    pub stem: String,
    pub card_id: String, // e.g. "CardId::Thunderclap"
    pub body: String,    // the `{ … }` block, verbatim
}

pub fn parse_specs(src: &str) -> Vec<SpecFactory> {
    let file = syn::parse_file(src).expect("specs file parses");
    let mut out = Vec::new();
    for item in &file.items {
        let syn::Item::Fn(f) = item else { continue };
        // Only `-> CardSpec` factories.
        let is_spec = matches!(&f.sig.output, syn::ReturnType::Type(_, t)
            if t.to_token_stream().to_string().replace(' ', "") == "CardSpec");
        if !is_spec {
            continue;
        }
        let stem = f.sig.ident.to_string();
        // Body span (the block braces), verbatim from source.
        let blk = &f.block;
        let r = blk.span().byte_range();
        let body = src[r.start..r.end].to_string();
        // CardId from the `id:` field inside the block.
        let card_id = find_id_field(blk);
        out.push(SpecFactory {
            stem,
            card_id,
            body,
        });
    }
    out
}

/// Extract the `id: CardId::X` field value from a CardSpec literal block.
fn find_id_field(block: &syn::Block) -> String {
    struct V(Option<String>);
    impl<'ast> syn::visit::Visit<'ast> for V {
        fn visit_field_value(&mut self, fv: &'ast syn::FieldValue) {
            if let syn::Member::Named(n) = &fv.member {
                if n == "id" && self.0.is_none() {
                    self.0 = Some(fv.expr.to_token_stream().to_string().replace(' ', ""));
                }
            }
            syn::visit::visit_field_value(self, fv);
        }
    }
    let mut v = V(None);
    syn::visit::Visit::visit_block(&mut v, block);
    v.0.expect("CardSpec literal has an `id:` field")
}

use std::collections::BTreeMap;

/// Map `"CardId::X"` -> the arm body expression text (e.g. `Some(CardUpgradeDelta { … })`).
pub fn parse_upgrade_arms(src: &str) -> BTreeMap<String, String> {
    let file = syn::parse_file(src).expect("upgrades file parses");
    let mut out = BTreeMap::new();
    for item in &file.items {
        let syn::Item::Fn(f) = item else { continue };
        if f.sig.ident != "upgrade_delta" {
            continue;
        }
        // Body is `{ match card_id { … } }`; find the match expr.
        for stmt in &f.block.stmts {
            let syn::Stmt::Expr(expr, _) = stmt else {
                continue;
            };
            if let syn::Expr::Match(m) = expr {
                for arm in &m.arms {
                    // Skip the `_ => None` wildcard.
                    if matches!(arm.pat, syn::Pat::Wild(_)) {
                        continue;
                    }
                    let key = arm.pat.to_token_stream().to_string().replace(' ', "");
                    // Arm body verbatim from source (preserves `Some(7)` spacing),
                    // matching parse_specs' verbatim relocation rather than
                    // re-rendering tokens via quote (which inserts `Some (7)`).
                    let r = arm.body.span().byte_range();
                    let val = src[r.start..r.end].to_string();
                    out.insert(key, val);
                }
            }
        }
    }
    out
}

use crate::split_cards::{raw_mod, rewrite_super};
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct Plan {
    /// stem -> text to append to (or create) cards/<char>/<stem>.rs
    pub appendix: BTreeMap<String, String>,
    /// stems with no existing handler file (emitted as spec-only files)
    pub spec_only: BTreeSet<String>,
    /// regenerated `<CHAR>_SPECS` slice element lines (in source order)
    pub specs_slice: String,
    /// regenerated delegation arms for this char's upgraded cards
    pub delegations: String,
}

pub fn build_plan(
    specs_src: &str,
    upgrades_src: &str,
    existing_stems: &BTreeSet<String>,
    character: &str,
    delegate_stems: &BTreeSet<String>,
) -> Plan {
    let specs = parse_specs(specs_src);
    let arms = parse_upgrade_arms(upgrades_src);

    let mut appendix = BTreeMap::new();
    let mut spec_only = BTreeSet::new();
    let mut specs_slice = String::new();
    let mut delegations = String::new();

    for s in &specs {
        // spec(): rename `<stem>()` to `spec()`, body verbatim (super:: rewritten).
        // For a handrolled-override stem (colorless's authoritative `splash`/
        // `panache_card`/… which the central slice used INSTEAD of the generated
        // duplicate, and which the pool's TESTS still call directly), `spec()`
        // delegates to the surviving handrolled `helpers::<stem>()` rather than
        // inlining the stale generated body. Byte-identical, single source.
        let body = if delegate_stems.contains(&s.stem) {
            format!("{{\n    super::helpers::{}()\n}}", raw_mod(&s.stem))
        } else {
            rewrite_super(&s.body)
        };
        let mut text = format!("\n#[must_use]\npub(crate) const fn spec() -> CardSpec {body}\n");

        // upgrade(): the arm body, or `None`. The relocated body always
        // returns `Some(CardUpgradeDelta { … })` (→ clippy::unnecessary_wraps)
        // or a bare `None` whose `upgrade()` is never delegated to when the
        // card has no upgrade arm (→ dead_code). Both are inherent to the
        // verbatim relocation, not logic to "fix" — suppress at the fn.
        let allow = "#[allow(clippy::unnecessary_wraps, dead_code)]\n";
        let up_body = arms.get(&s.card_id).map(|e| rewrite_super(e));
        match &up_body {
            Some(expr) => {
                text.push_str(&format!(
                    "\n{allow}pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {{\n    {expr}\n}}\n"
                ));
                delegations.push_str(&format!(
                    "        {} => crate::cards::{character}::{}::upgrade(),\n",
                    s.card_id,
                    raw_mod(&s.stem)
                ));
            }
            None => {
                text.push_str(&format!(
                    "\n{allow}pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {{\n    None\n}}\n"
                ));
            }
        }

        appendix.insert(s.stem.clone(), text);
        if !existing_stems.contains(&s.stem) {
            spec_only.insert(s.stem.clone());
        }
        specs_slice.push_str(&format!(
            "    crate::cards::{character}::{}::spec,\n",
            raw_mod(&s.stem)
        ));
    }

    Plan {
        appendix,
        spec_only,
        specs_slice,
        delegations,
    }
}

use std::path::Path;

pub fn run(character: &str) {
    let cards = Path::new("crates/sts2-content/src/cards");
    let specs_path = cards.join(format!("{character}_specs_generated.rs"));
    let upgrades_path = cards.join("card_upgrades_generated.rs");
    // Some pools (e.g. necrobinder, curse) have NO `<char>_specs_generated.rs`
    // — their card specs are not in a separate generated file, so there is
    // nothing to co-locate. No-op cleanly (the handler split already produced
    // per-card files).
    if !specs_path.exists() {
        eprintln!("colocate {character}: no {character}_specs_generated.rs — nothing to co-locate");
        return;
    }
    let specs_src = std::fs::read_to_string(&specs_path).expect("read specs");
    let upgrades_src = std::fs::read_to_string(&upgrades_path).expect("read upgrades");

    let dir = cards.join(character);
    // Existing handler stems on disk, both bare (e.g. `break`) and raw-escaped
    // (`r#break`) forms, so `build_plan`'s comparison against the syn-rendered
    // factory ident (which is `r#break` for keyword stems) matches.
    let mut existing: BTreeSet<String> = BTreeSet::new();
    for entry in std::fs::read_dir(&dir).expect("read char dir") {
        let Ok(entry) = entry else { continue };
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        let Some(stem) = name.strip_suffix(".rs") else {
            continue;
        };
        if matches!(stem, "mod" | "prelude" | "helpers" | "specs" | "tests") {
            continue;
        }
        existing.insert(stem.to_string());
        existing.insert(raw_mod(stem));
    }

    // Handrolled spec overrides: a pool may carry authoritative `-> CardSpec`
    // factories in helpers.rs that the central slice used INSTEAD of the
    // generated duplicate (colorless: splash/automation/fasten/panache_card/
    // prep_time/coordinate_card — panache_card's name differs, "PANACHE" vs the
    // generated "PANACHE_CARD"). Those handrolled fns must SURVIVE (the pool's
    // tests call them directly), so the co-located `spec()` delegates to them.
    let helpers_path = dir.join("helpers.rs");
    let delegate_stems = handrolled_override_stems(&helpers_path, &specs_src);

    let plan = build_plan(
        &specs_src,
        &upgrades_src,
        &existing,
        character,
        &delegate_stems,
    );

    // 1. Append spec()/upgrade() to each card file (create spec-only files).
    for (stem, text) in &plan.appendix {
        // On-disk file name strips the raw `r#` escape (`r#break` -> `break.rs`).
        let file_stem = stem.strip_prefix("r#").unwrap_or(stem);
        let path = dir.join(format!("{file_stem}.rs"));
        if plan.spec_only.contains(stem) {
            std::fs::write(
                &path,
                format!(
                    "//! @generated by xtask colocate-card-data (spec-only — handler lives elsewhere).\nuse super::prelude::*;\n{text}"
                ),
            )
            .expect("write spec-only");
        } else {
            let mut existing_text = std::fs::read_to_string(&path).expect("read card file");
            existing_text.push_str(text);
            std::fs::write(&path, existing_text).expect("append card file");
        }
    }

    // 1.5. Declare the newly-created spec-only modules in mod.rs. Handler
    // cards already have a `pub mod <stem>;` (split-cards wrote it); spec-only
    // starters (e.g. `bash`, `strike_ironclad`) had no handler file, so their
    // module must be added to the sorted decl block.
    insert_spec_only_mods(&dir, &plan.spec_only);

    // 2. Regenerate the <CHAR>_SPECS slice in spec_index.rs.
    regen_specs_slice(cards, character, &plan.specs_slice);

    // 3. Rewrite this char's upgrade_delta arms to delegations.
    rewrite_upgrade_arms(&upgrades_path, character, &plan);

    // 4. Re-point colorless cross-pool refs + names refs + this char's
    //    specs.rs sampler + drop the prelude `ic_specs` alias + mod.rs decl.
    repoint_references(cards, character, &plan);

    // 4.5. If handrolled specs were co-located (colorless), the pool's OWN
    //      module alias in spec_index (`colorless as co_handrolled`, which fed
    //      the slice's handrolled entries) is now unused — drop it.
    if !delegate_stems.is_empty() {
        let si_path = cards.join("spec_index.rs");
        let si = std::fs::read_to_string(&si_path).expect("read spec_index");
        std::fs::write(&si_path, drop_module_alias_import(&si, character))
            .expect("write spec_index");
    }

    // 5. Delete the now-empty generated specs file.
    std::fs::remove_file(&specs_path).expect("remove specs file");

    eprintln!(
        "colocated {character}: {} cards ({} spec-only)",
        plan.appendix.len(),
        plan.spec_only.len()
    );
}

/// Handrolled-override stems: `-> CardSpec` factories in `helpers.rs` whose
/// stem ALSO has a generated factory. The central slice used the handrolled
/// (authoritative) one — and the pool's tests call it directly — so the
/// co-located `spec()` delegates to `helpers::<stem>()` rather than inlining
/// the stale generated body. Empty if helpers.rs has no such factories (the
/// common case — only colorless has them).
fn handrolled_override_stems(helpers_path: &Path, generated_src: &str) -> BTreeSet<String> {
    let Ok(helpers_src) = std::fs::read_to_string(helpers_path) else {
        return BTreeSet::new();
    };
    let generated: BTreeSet<String> = parse_specs(generated_src)
        .into_iter()
        .map(|s| s.stem)
        .collect();
    parse_specs(&helpers_src)
        .into_iter()
        .map(|s| s.stem)
        .filter(|stem| generated.contains(stem))
        .collect()
}

/// Drop a `<character> as <alias>,` token (the pool's OWN module aliased — e.g.
/// colorless's `co_handrolled`, which fed the slice's handrolled-spec entries)
/// from spec_index's `use super::{ … }` import once the slice no longer uses
/// that alias. Matches only `<char> as …` (the import), never `<char>::…`
/// (the regenerated `crate::cards::<char>::<stem>::spec` slice paths).
fn drop_module_alias_import(src: &str, character: &str) -> String {
    let needle = format!("{character} as ");
    src.lines()
        .map(|line| {
            let Some(pos) = line.find(&needle) else {
                return line.to_string();
            };
            // Token boundary before `<character>` (not part of a longer ident).
            let bytes = line.as_bytes();
            let before_ok =
                pos == 0 || !(bytes[pos - 1].is_ascii_alphanumeric() || bytes[pos - 1] == b'_');
            if !before_ok {
                return line.to_string();
            }
            let after = &line[pos + needle.len()..];
            let alias_end = after
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(after.len());
            let rest = &after[alias_end..];
            let rest = rest.strip_prefix(',').unwrap_or(rest);
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            format!("{}{}", &line[..pos], rest)
        })
        .collect::<Vec<_>>()
        .join("\n")
        + if src.ends_with('\n') { "\n" } else { "" }
}

/// Insert `pub mod <stem>;` for each spec-only stem into the sorted module
/// declaration block in `<dir>/mod.rs`. Idempotent: skips stems already
/// declared. Keeps the `pub mod …;` run sorted to match split-cards' output.
fn insert_spec_only_mods(dir: &Path, spec_only: &BTreeSet<String>) {
    if spec_only.is_empty() {
        return;
    }
    let modrs = dir.join("mod.rs");
    let src = std::fs::read_to_string(&modrs).expect("read mod.rs");
    let mut lines: Vec<String> = src.lines().map(str::to_string).collect();

    // Card-stem decls are `pub mod <name>;` and form a sorted run, but the
    // block also contains a few non-card decls (`mod prelude;`, `#[cfg(test)]
    // mod tests;`) that must be left in place. So we insert each missing decl
    // at the correct sorted position *among the existing `pub mod` lines*,
    // touching nothing else.
    let is_pub_mod = |l: &str| {
        let t = l.trim_start();
        t.starts_with("pub mod ") && t.ends_with(';') && !t.contains('{')
    };
    for stem in spec_only {
        let decl = format!("pub mod {};", raw_mod(stem));
        if lines.iter().any(|l| l.trim() == decl) {
            continue; // already declared (idempotent)
        }
        // Insert before the first existing `pub mod` line that sorts after
        // `decl`; if none, append after the last `pub mod` line.
        let pos = lines
            .iter()
            .position(|l| is_pub_mod(l) && l.trim() > decl.as_str());
        if let Some(p) = pos {
            lines.insert(p, decl);
        } else {
            let last = lines
                .iter()
                .rposition(|l| is_pub_mod(l))
                .expect("no `pub mod` block in mod.rs");
            lines.insert(last + 1, decl);
        }
    }

    let mut text = lines.join("\n");
    if !text.ends_with('\n') {
        text.push('\n');
    }
    std::fs::write(&modrs, text).expect("write mod.rs");
}

/// Replace the `pub const <CHAR>_SPECS: &[fn() -> CardSpec] = &[ … ];` body.
fn regen_specs_slice(cards: &Path, character: &str, slice_body: &str) {
    let path = cards.join("spec_index.rs");
    let src = std::fs::read_to_string(&path).expect("read spec_index");
    let upper = character.to_uppercase();
    let marker = format!("{upper}_SPECS: &[fn() -> CardSpec] = &[");
    let start = src.find(&marker).expect("find SPECS slice") + marker.len();
    let end = start + src[start..].find("\n];").expect("find slice end");
    let mut out = String::with_capacity(src.len());
    out.push_str(&src[..start]);
    out.push('\n');
    out.push_str(slice_body.trim_end_matches('\n'));
    out.push_str(&src[end..]);
    std::fs::write(&path, out).expect("write spec_index");
}

/// Replace each `CardId::X => Some(…) ,` arm for this char with a delegation.
/// Original upgraded arms are identified by the CardIds present in
/// `plan.delegations`; we delete the char's original arms and insert the
/// delegation block before the `_ => None,` wildcard. Non-upgraded cards had
/// no arm to begin with, so per-card `upgrade()` returning `None` is correct.
fn rewrite_upgrade_arms(path: &Path, _character: &str, plan: &Plan) {
    let src = std::fs::read_to_string(path).expect("read upgrades");
    let owned: BTreeSet<String> = plan
        .delegations
        .lines()
        .filter_map(|l| l.trim().split(" =>").next().map(str::to_string))
        .collect();
    let lines: Vec<&str> = src.lines().collect();
    let mut kept: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim_start();
        let owned_arm = owned
            .iter()
            .any(|cid| trimmed.starts_with(&format!("{cid} =>")));
        if owned_arm {
            // Skip the whole arm. A one-line arm closes on its own first
            // line (`CardId::X => Some(…),` / `CardId::X => None,`); a
            // multi-line `Some(CardUpgradeDelta { … })` arm closes on the
            // line ending in `}),`. Inner field lines such as
            // `keywords: Some(CardKeywords::INNATE),` also end in `),`, so we
            // must NOT treat a bare `),` as the arm terminator — only `}),`
            // (or a self-terminating first line) closes the arm.
            let first = lines[i].trim_end();
            if first.ends_with("}),") || first.ends_with("None,") {
                // self-contained one-line arm
                i += 1;
                continue;
            }
            // multi-line arm: skip until the `}),` that closes it.
            while i < lines.len() && !lines[i].trim_end().ends_with("}),") {
                i += 1;
            }
            i += 1; // skip the closing `}),` line too
            continue;
        }
        kept.push(lines[i]);
        i += 1;
    }
    let mut rebuilt = kept.join("\n");
    rebuilt = rebuilt.replace(
        "        _ => None,",
        &format!("{}        _ => None,", plan.delegations),
    );
    if !rebuilt.ends_with('\n') {
        rebuilt.push('\n');
    }
    std::fs::write(path, rebuilt).expect("write upgrades");
}

/// Re-point colorless cross-pool refs, names refs, this char's specs.rs
/// sampler, drop the prelude `ic_specs` alias, and the mod.rs decl.
fn repoint_references(cards: &Path, character: &str, plan: &Plan) {
    let genmod = format!("{character}_specs_generated");
    // Set of spec stems (raw form) this char owns, for call-form fixups.
    let stems: BTreeSet<String> = plan.appendix.keys().cloned().collect();

    // colorless.rs: `super::super::<genmod>::<stem>()` spec calls. This fixes
    // the (still-monolithic) colorless pool's cross-pool references to whatever
    // pool is being colocated. Once colorless ITSELF has been split, the
    // monolith is gone — its self-references go through the `cspecs` prelude
    // alias and are handled by the tree-alias rewrite below, so skip it.
    let colorless_path = cards.join("colorless.rs");
    if colorless_path.exists() {
        let src = std::fs::read_to_string(&colorless_path).expect("read colorless");
        let out = fixup_colorless_calls(&src, character, &genmod, &stems);
        std::fs::write(&colorless_path, out).expect("write colorless");
    }

    // spec_index.rs: the IRONCLAD_SPECS slice is already regenerated; here we
    // only drop the `<genmod> as ic` import token from the `use super::{…}`.
    let spec_index_path = cards.join("spec_index.rs");
    let si = std::fs::read_to_string(&spec_index_path).expect("read spec_index");
    std::fs::write(&spec_index_path, drop_genmod_import(&si, &genmod)).expect("write spec_index");

    // Per-card tree (helpers.rs, specs.rs, tests.rs, every <card>.rs): the
    // monolith referenced its generated specs through a prelude alias
    // (`ic_specs` for Ironclad, `specs` for Silent, …). Discover that alias
    // from the prelude, then re-point `<alias>::<stem>` ->
    // `crate::cards::<char>::<stem>::spec` everywhere in the tree, and drop any
    // dangling `use super::prelude::<alias>;` disambiguator (the alias is about
    // to be removed from the prelude). Keyed off the discovered alias, so it is
    // pool-agnostic.
    let char_dir = cards.join(character);
    let prelude_src = std::fs::read_to_string(char_dir.join("prelude.rs")).expect("read prelude");
    let tree_alias = find_genmod_alias(&prelude_src, &genmod);
    if let Some(alias) = &tree_alias {
        let drop_line = format!("use super::prelude::{alias};");
        for entry in std::fs::read_dir(&char_dir)
            .expect("read char dir")
            .flatten()
        {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            if path.file_name().is_some_and(|n| n == "prelude.rs") {
                continue; // the alias import line is dropped below
            }
            let src = std::fs::read_to_string(&path).expect("read tree file");
            let out = src
                .lines()
                .filter(|l| l.trim() != drop_line)
                .map(|line| {
                    if line.trim_start().starts_with("///") {
                        line.to_string()
                    } else {
                        rewrite_alias_refs(line, alias, character)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
                + if src.ends_with('\n') { "\n" } else { "" };
            std::fs::write(&path, out).expect("write tree file");
        }
    }

    // prelude.rs: drop the `pub(crate) use … as ic_specs;` line, and bring
    // into scope the spec/upgrade types the relocated `spec()`/`upgrade()`
    // bodies reference. The old generated file imported these directly
    // (`TargetType` from `sts2_core::cards`; `CardKeywords`/`CardRarity` from
    // `crate::cards::types`); now that the bodies live in per-card files that
    // `use super::prelude::*`, the prelude must re-export them. `CardSpec`,
    // `CardId`, `CardType` are already in the prelude.
    let prelude = cards.join(character).join("prelude.rs");
    let pl = std::fs::read_to_string(&prelude).expect("read prelude");
    std::fs::write(&prelude, fixup_prelude(&pl, &genmod)).expect("write prelude");

    // names.rs: `<genmod> as ic` import + `ic::<stem>` refs.
    let names = cards.parent().unwrap().join("names.rs");
    let n = std::fs::read_to_string(&names).expect("read names");
    std::fs::write(&names, fixup_names(&n, character, &genmod)).expect("write names");

    // mod.rs: drop the module declaration.
    let modrs = cards.join("mod.rs");
    let m = std::fs::read_to_string(&modrs).expect("read mod.rs");
    let m: String = m
        .lines()
        .filter(|l| !l.contains(&format!("mod {genmod};")))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&modrs, format!("{m}\n")).expect("write mod.rs");

    // <char>/tests.rs: split-cards emits glob imports of every per-card
    // module (`aggression::*, …`). Now that each module exports `spec()` and
    // `upgrade()`, the name `spec` (and `upgrade`) is glob-imported from all
    // 86 modules → E0659 ("ambiguous because of multiple glob imports") at any
    // bare use such as `for spec in &pool`. An explicit single-source import
    // takes precedence over glob imports, collapsing the ambiguity; in pattern
    // position a fresh `for spec` binding still shadows it. Idempotent.
    let tests_rs = cards.join(character).join("tests.rs");
    if let Ok(t) = std::fs::read_to_string(&tests_rs) {
        std::fs::write(&tests_rs, disambiguate_tests_globs(&t)).expect("write tests.rs");
    }
}

/// Inject an explicit `use super::specs::{spec, upgrade};`-style disambiguation
/// into a `tests.rs` whose per-card glob imports now collide on `spec`/`upgrade`.
/// We bind both names to a single source (the first per-card module glob in the
/// import list) so the multi-glob ambiguity collapses. Idempotent.
pub fn disambiguate_tests_globs(src: &str) -> String {
    const MARKER: &str = "// @colocate-disambiguate";
    if src.contains(MARKER) {
        return src.to_string();
    }
    // Pick the first per-card module named in the `use super::{ … };` glob
    // block (e.g. `aggression`); it exports both `spec` and `upgrade`.
    // NB: validity check is INSIDE find_map so a multi-line glob — where the
    // `use super::{` opener line yields an empty segment (the modules wrap onto
    // the next line after cargo fmt) — keeps scanning to the first real module
    // instead of bailing to the fallback. (defect's glob was single-line at
    // colocate time; colorless's was fmt-wrapped, exposing this.)
    let first_mod = src
        .lines()
        .filter_map(|l| {
            let t = l.trim_start();
            t.strip_prefix("use super::{")
                .or_else(|| if t.contains("::*,") { Some(t) } else { None })
        })
        .find_map(|seg| {
            let m = seg.split("::*").next().map(str::trim)?;
            (!m.is_empty() && m.chars().all(|c| c.is_alphanumeric() || c == '_'))
                .then(|| m.to_string())
        })
        .unwrap_or_else(|| "aggression".to_string());
    // Insert after the first `use super::prelude::*;` line (always present in
    // split-cards' tests.rs).
    let inject = format!(
        "{MARKER}: collapse multi-glob `spec`/`upgrade` ambiguity (E0659).\n#[allow(unused_imports)]\nuse super::{first_mod}::{{spec, upgrade}};\n"
    );
    if let Some(pos) = src.find("use super::prelude::*;\n") {
        let cut = pos + "use super::prelude::*;\n".len();
        let mut out = String::with_capacity(src.len() + inject.len());
        out.push_str(&src[..cut]);
        out.push_str(&inject);
        out.push_str(&src[cut..]);
        out
    } else {
        format!("{inject}{src}")
    }
}

/// `super::super::<genmod>::<stem>()` -> `crate::cards::<char>::<stem>::spec()`.
fn fixup_colorless_calls(
    src: &str,
    character: &str,
    genmod: &str,
    _stems: &BTreeSet<String>,
) -> String {
    let needle = format!("super::super::{genmod}::");
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(pos) = rest.find(&needle) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + needle.len()..];
        // Read the stem ident up to `(`.
        let paren = after.find('(').expect("spec call has parens");
        let stem = &after[..paren];
        out.push_str(&format!("crate::cards::{character}::{stem}::spec"));
        rest = &after[paren..];
    }
    out.push_str(rest);
    out
}

/// Drop the `<genmod> as <alias>` token from a `use super::{ … };` import block.
fn drop_genmod_import(src: &str, genmod: &str) -> String {
    src.lines()
        .map(|line| {
            if !line.contains(genmod) {
                return line.to_string();
            }
            // Remove the `<genmod> as <alias>,` token (with optional trailing
            // comma + whitespace) from the brace list.
            let mut cleaned = String::with_capacity(line.len());
            let mut rest = line;
            while let Some(pos) = rest.find(genmod) {
                cleaned.push_str(&rest[..pos]);
                let after = &rest[pos + genmod.len()..];
                // Skip ` as <alias>` then a trailing `, ` if present.
                let after = after.trim_start();
                let after = after.strip_prefix("as ").unwrap_or(after);
                // Skip the alias ident.
                let alias_end = after
                    .find(|c: char| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(after.len());
                let after = &after[alias_end..];
                let after = after.strip_prefix(',').unwrap_or(after);
                let after = after.strip_prefix(' ').unwrap_or(after);
                rest = after;
            }
            cleaned.push_str(rest);
            cleaned
        })
        .collect::<Vec<_>>()
        .join("\n")
        + if src.ends_with('\n') { "\n" } else { "" }
}

/// Rewrite the per-card `prelude.rs` after the generated specs module is
/// deleted: drop every line that re-exports that module (whatever it was
/// aliased to — `as ic_specs`, `as specs`, …, keyed alias-agnostically on the
/// genmod name) plus any attribute lines (e.g. a lone `#[cfg(test)]`) directly
/// preceding a dropped line, then idempotently re-export the spec/upgrade-body
/// type imports the relocated `spec()`/`upgrade()` bodies need.
///
/// Two defects this guards against (both first hit on the defect pool):
/// 1. Relocated bodies reference `CardSpec`, but the split-generated prelude
///    only carries what the HANDLER bodies imported; pools whose handlers
///    never name `CardSpec` (defect) lacked it.
/// 2. Dropping only the `use … as specs;` line orphaned its preceding lone
///    `#[cfg(test)]`, which re-bound to the next surviving `use` (the shared
///    `status::{…}` import), wrongly gating those fns behind test cfg.
pub fn fixup_prelude(pl: &str, genmod: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    for l in pl.lines() {
        if l.contains(genmod) {
            while kept
                .last()
                .is_some_and(|p| p.trim_start().starts_with("#["))
            {
                kept.pop();
            }
            continue;
        }
        kept.push(l);
    }
    let mut pl: String = kept.join("\n");
    // The relocated spec() bodies reference `CardSpec`, `CardId`, `CardType`,
    // the keyword/rarity/delta types, and `TargetType`. The split prelude only
    // carries what the HANDLER bodies imported, so any of these used ONLY in
    // spec bodies is absent for some pools (CardSpec for defect; CardType for
    // regent, whose handlers never name it). Add each missing piece.
    if !pl.contains("CardSpec") {
        pl.push_str("\npub(crate) use crate::cards::types::CardSpec;");
    }
    if !pl.contains("CardType") {
        pl.push_str("\npub(crate) use sts2_core::cards::CardType;");
    }
    if !pl.contains("CardId") {
        pl.push_str("\npub(crate) use sts2_core::cards::CardId;");
    }
    if !pl.contains("cards::types::CardUpgradeDelta") {
        pl.push_str(
            "\npub(crate) use crate::cards::types::{CardKeywords, CardRarity, CardUpgradeDelta};",
        );
    }
    if !pl.contains("TargetType") {
        pl.push_str("\npub(crate) use sts2_core::cards::TargetType;");
    }
    format!("{pl}\n")
}

/// names.rs: drop `<genmod> as ic` import token + re-point `ic::<stem>` factory
/// refs to `crate::cards::<char>::<stem>::spec`.
fn fixup_names(src: &str, character: &str, genmod: &str) -> String {
    // The short alias differs per pool (`ic`, `si`, `de`, …); discover it from
    // the `<genmod> as <alias>` import rather than hardcoding `ic`.
    let alias = find_genmod_alias(src, genmod).unwrap_or_else(|| genmod.to_string());
    let dropped = drop_genmod_import(src, genmod);
    // Re-point `<alias>::<stem>` -> `crate::cards::<char>::<stem>::spec`, but
    // NOT inside doc comments (those start with `///`).
    dropped
        .lines()
        .map(|line| {
            if line.trim_start().starts_with("///") {
                return line.to_string();
            }
            rewrite_alias_refs(line, &alias, character)
        })
        .collect::<Vec<_>>()
        .join("\n")
        + if dropped.ends_with('\n') { "\n" } else { "" }
}

/// Find the local alias of `<genmod>` from a `… <genmod> as <alias>` import.
/// Returns `None` if the module is imported unaliased (use the genmod name).
fn find_genmod_alias(src: &str, genmod: &str) -> Option<String> {
    let needle = format!("{genmod} as ");
    let pos = src.find(&needle)? + needle.len();
    let after = &src[pos..];
    let end = after
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(after.len());
    Some(after[..end].to_string())
}

/// `<alias>::<stem>` -> `crate::cards::<char>::<stem>::spec` on a single line.
fn rewrite_alias_refs(line: &str, alias: &str, character: &str) -> String {
    let needle = format!("{alias}::");
    let needle = needle.as_str();
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(pos) = rest.find(needle) {
        // Guard: only match `<alias>::` as a standalone path head — the char
        // before it must not be an ident char OR `:` (avoids `magic::` and
        // path-internal `super::specs::*`).
        let preceding_ok = pos == 0
            || !rest[..pos]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_' || c == ':');
        let after = &rest[pos + needle.len()..];
        let stem_end = after
            .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '#')
            .unwrap_or(after.len());
        let stem = &after[..stem_end];
        // Skip non-path-head matches and glob/empty stems (`<alias>::*`).
        if !preceding_ok || stem.is_empty() {
            let keep = pos + needle.len();
            out.push_str(&rest[..keep]);
            rest = &rest[keep..];
            continue;
        }
        out.push_str(&rest[..pos]);
        out.push_str(&format!("crate::cards::{character}::{stem}::spec"));
        rest = &after[stem_end..];
    }
    out.push_str(rest);
    out
}
