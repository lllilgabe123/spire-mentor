//! The built-in PUCT move-prior nets (`sts2_mcts::rollout_net::builtin_prior_json`,
//! `data/prior_nets/<character>.json`, GH #830). Only Ironclad has one: the
//! behaviour-cloned prior of the bundle that passed its full-run gate
//! (Ironclad A10, PR #938).
//!
//! These tests pin the gated file itself, not "whatever is committed": a new
//! prior net needs a new full-run gate.

use sha2::{Digest, Sha256};
use sts2_core::CharacterId;
use sts2_mcts::rollout_net::{builtin_prior_json, RolloutNet};

/// sha256 of the shipped `data/prior_nets/ironclad.json`: the gated net
/// (`bc_ironclad_v2.json`, sha256 `95f3e909…3651`) with its `meta` set to `{}`
/// and nothing else changed.
const SHIPPED_IRONCLAD_PRIOR_SHA256: &str =
    "ca4692595c22d1c26741c4557e1f7878151e253129ff2e8779d59269624ad4f2";

fn install() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
}

/// Ironclad has a built-in prior exactly when its file is in the checkout,
/// and it parses. No other character has one, whatever files exist: a net
/// dropped into `data/prior_nets/` must not switch another character on.
#[test]
fn built_in_prior_nets_load() {
    install();
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/prior_nets/ironclad.json");
    let text = builtin_prior_json(CharacterId::Ironclad);
    assert_eq!(text.is_some(), file.is_file(), "Ironclad");
    if let Some(text) = text {
        let net = RolloutNet::from_json(text).expect("the built-in Ironclad prior parses");
        assert!(net.rows > 100, "{} embedding rows", net.rows);
    }
    for c in [
        CharacterId::Silent,
        CharacterId::Defect,
        CharacterId::Regent,
        CharacterId::Necrobinder,
    ] {
        assert!(
            builtin_prior_json(c).is_none(),
            "{c:?} has a built-in prior"
        );
    }
}

/// Not cfg-gated: a checkout, container or export without
/// `data/prior_nets/ironclad.json` still applies the Ironclad knobs (compiled
/// code, `IRONCLAD_SEARCH_KNOBS` in `sts2-runai`) but searches without the
/// move prior. That is an ungated knobs-only search, not the gate's control
/// or its bundle, and this test makes it fail loudly instead of playing it
/// silently. The asserted value is constant per build on purpose: it is the
/// build's own cfg.
#[test]
#[allow(clippy::assertions_on_constants)]
fn the_ironclad_prior_net_is_compiled_in() {
    assert!(
        cfg!(prior_net_ironclad),
        "data/prior_nets/ironclad.json is missing: the Ironclad run agent would search without its move prior (GH #830)"
    );
}

#[test]
fn the_built_in_ironclad_prior_is_the_shipped_gated_net() {
    let text = builtin_prior_json(CharacterId::Ironclad).expect("compiled in");
    let sha = format!("{:x}", Sha256::digest(text.as_bytes()));
    assert_eq!(
        sha, SHIPPED_IRONCLAD_PRIOR_SHA256,
        "the gated net is 95f3e909… with meta stripped; a new prior net needs a new full-run gate (GH #830)"
    );
}
