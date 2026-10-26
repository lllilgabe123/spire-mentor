//! The interactive surfaces search in the run agent's base dialect (GH
//! #758), and the run agent's per-character layer is exactly what it adds
//! on top (GH #830).
//!
//! The real-game advisor, the workbench hint / inspector / coach and SimLab
//! build their search from `sts2_wire::emit::mcts::RecommendParams`; the run
//! agent that every published win rate and promotion gate measured builds
//! it in `sts2_runai::policy::mcts_config_for`. PR #467 retuned the EndTurn
//! widening in the second only, and the first kept searching at 0.5 / 0.4
//! for two months without any test noticing. Both now call
//! `MctsConfig::apply_mode_dialect`; this test fails the next time either
//! side grows a dialect field the other does not have.
//!
//! On top of that base the run agent adds a per-character layer
//! (`MctsCombatPolicy::search_config`): for Ironclad the gated knobs
//! `IRONCLAD_SEARCH_KNOBS` (GH #830), which the advisor deliberately does
//! not take. The tests below pin that the layer is exactly those three
//! knobs, and that no other character has one.
//!
//! It lives here because this is the one crate that depends on both
//! `sts2-wire` and `sts2-runai`.

use std::path::PathBuf;

use serde_json::{Map, Value};
use sts2_core::potions::CharacterId;
use sts2_mcts::{HiddenInfoMode, MctsConfig};
use sts2_runai::policy::{
    mcts_config_for, CombatBudget, FirstOptionPolicy, MctsCombatPolicy, COMBAT_ROLLOUT_DEPTH,
};
use sts2_wire::emit::mcts::RecommendParams;

/// Every field of the search DIALECT — what a mode deploys, as opposed to
/// the per-call budget, seed and rollout depth. Compared bit-exactly.
fn dialect(cfg: &MctsConfig) -> Vec<(&'static str, String)> {
    let f = |x: f32| format!("{:#010x}", x.to_bits());
    vec![
        ("hidden_info", format!("{:?}", cfg.hidden_info)),
        ("c_puct", f(cfg.c_puct)),
        ("chance_widening_c", f(cfg.chance_widening_c)),
        ("chance_widening_alpha", f(cfg.chance_widening_alpha)),
        ("draw_widening_c", f(cfg.draw_widening_c)),
        ("draw_widening_alpha", f(cfg.draw_widening_alpha)),
        ("endturn_widening_c", f(cfg.endturn_widening_c)),
        ("endturn_widening_alpha", f(cfg.endturn_widening_alpha)),
        (
            "use_transposition_table",
            cfg.use_transposition_table.to_string(),
        ),
        (
            "regularized_pi_temperature",
            f(cfg.regularized_pi_temperature),
        ),
        ("use_puct_prior", cfg.use_puct_prior.to_string()),
        // A loss-band term, not a widening constant, but the advisor and the
        // run agent must agree on it all the same (the GH #758 class).
        ("removed_counts_forms", cfg.removed_counts_forms.to_string()),
    ]
}

fn run_agent(sims: u32) -> MctsConfig {
    mcts_config_for(
        HiddenInfoMode::BeliefDraw,
        sims,
        0,
        COMBAT_ROLLOUT_DEPTH,
        MctsConfig::default().terminal_eval_mode,
    )
}

fn assert_same_dialect(label: &str, surface: &MctsConfig, agent: &MctsConfig) {
    assert_eq!(
        dialect(surface),
        dialect(agent),
        "{label}: the interactive search's dialect drifted from the run agent's \
         (`mcts_config_for`). Both must come from `MctsConfig::apply_mode_dialect`."
    );
}

#[test]
fn recommend_params_default_matches_the_run_agent() {
    let p = RecommendParams::default();
    assert_same_dialect(
        "RecommendParams::default()",
        &p.mcts_config(),
        &run_agent(p.sims),
    );
}

/// A config that names only the budget gets the constants the run agent
/// would use at that budget (`c_puct` is budget-keyed).
#[test]
fn recommend_params_match_the_run_agent_at_every_budget() {
    for sims in [1, 300, 499, 500, 999, 1000, 2000, 4000, 8000] {
        let p =
            RecommendParams::from_config(&serde_json::json!({ "sims": sims }), "parity").unwrap();
        assert_same_dialect(&format!("sims={sims}"), &p.mcts_config(), &run_agent(sims));
    }
}

/// The budgets the interactive surfaces search at, and around the
/// `c_puct` steps.
const BUDGETS: [u32; 9] = [1, 300, 499, 500, 999, 1000, 2000, 4000, 8000];

/// The run agent's own config for a `character` fight at `sims`
/// (`MctsCombatPolicy::search_config`: the base plus its character layer).
fn run_agent_for(character: Option<CharacterId>, sims: u32) -> MctsConfig {
    MctsCombatPolicy::new(FirstOptionPolicy, CombatBudget::eval_cheap(), 0)
        .search_config(sims, 0, character)
        .0
}

/// The interactive search at `sims`.
fn interactive(sims: u32) -> MctsConfig {
    RecommendParams::from_config(&serde_json::json!({ "sims": sims }), "parity")
        .unwrap()
        .mcts_config()
}

/// The dialect fields where `a` and `b` differ.
fn differing(a: &MctsConfig, b: &MctsConfig) -> Vec<&'static str> {
    dialect(a)
        .into_iter()
        .zip(dialect(b))
        .filter(|((_, x), (_, y))| x != y)
        .map(|((k, _), _)| k)
        .collect()
}

/// GH #830: the Ironclad run agent differs from the interactive search in
/// exactly the three gated knobs, at every budget. A future advisor
/// extension (or a fourth run-agent knob) has to update this on purpose.
#[test]
fn run_agent_ironclad_differs_from_the_interactive_search_in_exactly_the_bundle_knobs() {
    for sims in BUDGETS {
        let mut got = differing(
            &run_agent_for(Some(CharacterId::Ironclad), sims),
            &interactive(sims),
        );
        got.sort_unstable();
        assert_eq!(
            got,
            [
                "draw_widening_c",
                "endturn_widening_c",
                "removed_counts_forms"
            ],
            "sims={sims}"
        );
    }
}

/// Every other character, and a fight without one, searches in the run
/// agent exactly as the interactive surfaces do.
#[test]
fn run_agent_other_characters_match_the_interactive_search() {
    for sims in BUDGETS {
        for character in [
            Some(CharacterId::Silent),
            Some(CharacterId::Defect),
            Some(CharacterId::Regent),
            Some(CharacterId::Necrobinder),
            None,
        ] {
            assert_same_dialect(
                &format!("{character:?} sims={sims}"),
                &interactive(sims),
                &run_agent_for(character, sims),
            );
        }
    }
}

/// The SPA's `DEFAULT_MCTS_CONFIG` body — what the browser hint, "play
/// best", autoplay and the inspector send — searches like the run agent at
/// its budget. Read from the source, so editing the SPA default is covered.
#[test]
fn spa_default_mcts_config_matches_the_run_agent() {
    let body = spa_default_mcts_config();
    let sims = u32::try_from(body["sims"].as_u64().expect("sims")).unwrap();
    let p = RecommendParams::from_config(&body, "parity").unwrap();
    assert_same_dialect(
        "web DEFAULT_MCTS_CONFIG",
        &p.mcts_config(),
        &run_agent(sims),
    );
}

/// Parse the `DEFAULT_MCTS_CONFIG` object literal out of `web/src/api.ts`.
/// Its entries are one `key: number | "string",` per line; comments are
/// dropped. Anything else fails loudly rather than parsing partially.
fn spa_default_mcts_config() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../sts2-pycli/web/src/api.ts");
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let start = src
        .find("export const DEFAULT_MCTS_CONFIG: MctsConfig = {")
        .expect("DEFAULT_MCTS_CONFIG not found in api.ts");
    let body = &src[start..];
    let body = &body[body.find('{').unwrap() + 1..body.find("\n};").expect("closing `};`")];
    let mut out = Map::new();
    for line in body.lines() {
        let line = line.split("//").next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let (key, val) = line
            .trim_end_matches(',')
            .split_once(':')
            .unwrap_or_else(|| panic!("unparseable DEFAULT_MCTS_CONFIG line: {line:?}"));
        let val: Value = serde_json::from_str(val.trim())
            .unwrap_or_else(|e| panic!("DEFAULT_MCTS_CONFIG value {val:?}: {e}"));
        out.insert(key.trim().to_string(), val);
    }
    assert!(out.contains_key("hidden_info"), "parsed {out:?}");
    Value::Object(out)
}
