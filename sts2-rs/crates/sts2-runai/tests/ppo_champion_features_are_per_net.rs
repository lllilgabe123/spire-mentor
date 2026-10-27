// Needs the ONNX runtime (`ppo::net`), compiled out without the `ppo` feature.
#![cfg(feature = "ppo")]
//! A committed champion is served at ITS OWN encoder schema, whatever
//! `STS2_PPO_FEATURES` says (GH #545).
//!
//! ## The bug this pins
//!
//! `STS2_PPO_FEATURES` is parsed **once per process**
//! ([`Features::active`]'s `OnceLock`). One process serving all five
//! characters therefore cannot express "on for Ironclad, off for the other
//! four" — and that is exactly the state the #545 campaign creates: it trains
//! Ironclad with `reward_upgrade` while the four incumbents were trained
//! without it.
//!
//! Handing an incumbent that feature is not a cosmetic mismatch. Every
//! champion to date holds an unconditional anti-upgraded-card prior
//! (`wb-27d55ac0`) learned from a hardcoded `upgraded=false` on the
//! most-exercised surface in training. Switch the truth on underneath it and
//! the prior runs BACKWARDS on real upgraded rewards.
//!
//! ## Why an ordinary encoder test cannot catch it
//!
//! A gated feature rewrites an **existing** `cat` slot rather than appending a
//! column, so the tensor shapes are identical either way and neither consumer
//! can detect the mismatch from the graph it loaded — unlike the `emb_boss` /
//! `N_CONT` appends, which are self-describing. Nothing crashes, nothing warns;
//! the net is simply evaluated on an observation it never saw. That is why the
//! assertion below is on the resolved `Features`, not on a shape.
//!
//! This test binary sets the env var **process-wide and hostilely**, which is
//! safe only because nextest gives each test binary its own process. Under
//! plain `cargo test` it would leak into every other test in the shared
//! process — one more reason this repo mandates nextest.

use sts2_core::potions::CharacterId;
use sts2_runai::ppo::features::Features;
use sts2_runai::ppo::net::features_for;

const EVERY_CHARACTER: [CharacterId; 5] = [
    CharacterId::Ironclad,
    CharacterId::Silent,
    CharacterId::Defect,
    CharacterId::Regent,
    CharacterId::Necrobinder,
];

/// The champions `champions.toml` records with `features = ""`.
///
/// Kept separate from [`EVERY_CHARACTER`] because a hostile
/// `STS2_PPO_FEATURES=reward_upgrade` cannot discriminate for a champion that
/// legitimately declares that same feature. Ironclad left this list on
/// 2026-08-28 (`ppo_fullrun_ironclad_underdocks_it200`); Silent left it on
/// 2026-09-08 (`ppo_fullrun_silent_cont_it300`).
const DEFAULT_SCHEMA_CHAMPIONS: [CharacterId; 5] = [
    CharacterId::Defect,
    CharacterId::Regent,
    CharacterId::Necrobinder,
    // Moved here 2026-09-13 (GH #723). Both declared `reward_upgrade` until
    // then, and neither was ever TRAINED with it: the trainer's acting
    // checkpoint carried no cfg, so `NetPolicy` fell back to the default
    // schema and collected every trajectory with the feature off. Serving
    // them on meant feeding two nets a column they had never seen.
    CharacterId::Ironclad,
    CharacterId::Silent,
];

/// The champions `champions.toml` records with `features = "reward_upgrade"`.
///
/// **Empty since 2026-09-13** — see the note above. While it is empty the
/// silent-env test's positive direction is carried by
/// [`an_explicit_model_override_still_reads_the_environment`], which exercises
/// the same `Features::parse` path through the override lane. Repopulating
/// this list restores the stronger end-to-end check, and is the point at which
/// the pair below regains its full strength.
const REWARD_UPGRADE_CHAMPIONS: [CharacterId; 0] = [];

/// Neither list may go empty, and together they must still cover every
/// character.
///
/// This is what keeps the pair of tests below from going VACUOUS as champions
/// turn over. Each list is the discriminating set for one direction: if every
/// champion ends up declaring `reward_upgrade`, the hostile-env test has
/// nothing left to prove and would pass by iterating an empty slice; if none
/// does, the silent-env test loses its positive case the same way. Losing a
/// character from both lists would drop it from coverage entirely without
/// failing anything.
///
/// A promotion that empties one side is not a lint to silence — it means the
/// property genuinely can no longer be tested this way and the test needs a
/// different exemplar (an explicit override net, say), so it fails loudly here.
fn assert_the_lists_still_discriminate() {
    assert!(
        !DEFAULT_SCHEMA_CHAMPIONS.is_empty(),
        "no champion declares the default schema any more, so the hostile-env \
         test below can no longer tell 'the manifest won' from 'the env leaked'"
    );
    // NOTE: `REWARD_UPGRADE_CHAMPIONS` is deliberately allowed to be empty
    // (GH #723 emptied it). An empty list costs the silent-env test its
    // positive case, which is why the doc comment there now points at the
    // override-lane test instead of claiming coverage it no longer has. It is
    // NOT allowed to go empty silently, so the exemption is spelled out here
    // rather than expressed as a missing assertion.
    for character in EVERY_CHARACTER {
        assert!(
            DEFAULT_SCHEMA_CHAMPIONS.contains(&character)
                || REWARD_UPGRADE_CHAMPIONS.contains(&character),
            "{character:?} is in neither list, so nothing below covers it"
        );
    }
}

/// Set the process-wide env to the most hostile value there is, then assert
/// the manifest still wins for every champion the manifest says is
/// default-schema.
///
/// **[`REWARD_UPGRADE_CHAMPIONS`] are deliberately absent.** Their champions
/// genuinely declare `reward_upgrade` (Ironclad since 2026-08-28, Silent since
/// 2026-09-08), so under this env value "the manifest won" and "the env
/// leaked" produce the same answer and asserting it here would prove nothing.
/// The discriminating direction for them is
/// [`the_manifest_supplies_a_feature_the_environment_never_mentions`], which
/// runs with the env SILENT — between them the two pin env-independence for
/// all five.
///
/// One test function per env value, not five: the var is process-global and
/// `Features::active` memoises on first read. That is safe only because
/// nextest runs each test in its own process; under plain `cargo test` these
/// would corrupt each other, which is one more reason this repo mandates it.
#[test]
fn a_committed_champion_ignores_a_hostile_process_wide_feature_flag() {
    std::env::set_var("STS2_PPO_FEATURES", "reward_upgrade");
    // `features_for` hands the env var through ONLY for an explicit
    // `STS2_RUNAI_PPO` override, so the committed-champion path must be
    // exercised with that unset — otherwise this test would pass for the
    // wrong reason.
    std::env::remove_var("STS2_RUNAI_PPO");

    assert_the_lists_still_discriminate();

    // The hostile value really is live: if this failed, the assertions below
    // would be vacuous.
    assert_eq!(
        Features::active(),
        Features {
            reward_upgrade: true,
            card_mods: false,
        },
        "the env var this test exists to defeat is not actually set"
    );

    for character in DEFAULT_SCHEMA_CHAMPIONS {
        assert_eq!(
            features_for(character),
            Features::default(),
            "{character:?}'s committed champion was trained at the default \
             schema, but resolved to a feature set from the environment. A \
             process-wide flag must never reach a champion the manifest \
             describes — see this file's header for why nothing downstream \
             would notice."
        );
    }
}

/// The other direction: with the environment SILENT, every champion must
/// resolve to exactly the schema the manifest records for it.
///
/// It fails if a manifest value is ever dropped on the floor — the
/// mirror-image bug of a leaking env var, and the one that would serve a
/// `reward_upgrade` net on default-schema observations. Same silent failure,
/// opposite cause: identical graph, identical shapes, a prior running
/// backwards on upgraded card rewards.
///
/// **Since GH #723 no champion declares a feature**, so the
/// `REWARD_UPGRADE_CHAMPIONS` loop below is currently a no-op and the positive
/// direction — "a declared feature survives to the serving call" — is carried
/// by [`an_explicit_model_override_still_reads_the_environment`]. What remains
/// here is still load-bearing: it is what would catch the env leaking into all
/// five now that the default schema is unanimous.
///
/// If a future promotion changes which champions declare features, this
/// assertion is *supposed* to fail — it is a prompt to revisit both halves,
/// not a line to quietly update. `just check-champions` pins the generated
/// table itself; this pins that the table is what gets served.
#[test]
fn the_manifest_supplies_a_feature_the_environment_never_mentions() {
    std::env::remove_var("STS2_PPO_FEATURES");
    std::env::remove_var("STS2_RUNAI_PPO");

    assert_the_lists_still_discriminate();

    assert_eq!(
        Features::active(),
        Features::default(),
        "the environment is not actually silent, so this test cannot show \
         that the manifest is what supplied the feature below"
    );

    for character in REWARD_UPGRADE_CHAMPIONS {
        assert_eq!(
            features_for(character),
            Features {
                reward_upgrade: true,
                card_mods: false,
            },
            "{character:?}'s champion declares features=\"reward_upgrade\" in \
             champions.toml, but resolved to the default schema with the \
             environment silent — the manifest value is being dropped, and the \
             net would be served observations it never trained on."
        );
    }

    for character in DEFAULT_SCHEMA_CHAMPIONS {
        assert_eq!(
            features_for(character),
            Features::default(),
            "{character:?} declares no features and the environment is \
             silent, so it must resolve to the default schema"
        );
    }
}

/// The escape hatch still works: an explicitly overridden net has no manifest
/// entry, so `STS2_PPO_FEATURES` is the only way to declare its schema. This
/// is the lane a gate sweep of a not-yet-promoted champion runs in.
#[test]
fn an_explicit_model_override_still_reads_the_environment() {
    std::env::set_var("STS2_PPO_FEATURES", "reward_upgrade");
    std::env::set_var("STS2_RUNAI_PPO", "/nonexistent/candidate.onnx");

    for character in EVERY_CHARACTER {
        assert_eq!(
            features_for(character),
            Features {
                reward_upgrade: true,
                card_mods: false,
            },
            "{character:?}: with STS2_RUNAI_PPO pointing at a net no manifest \
             describes, STS2_PPO_FEATURES is the only schema declaration there \
             is and must be honoured"
        );
    }
}
