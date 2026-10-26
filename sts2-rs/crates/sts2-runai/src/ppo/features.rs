//! Opt-in encoder experiment features — the Rust mirror of `ppo/schema.py`'s
//! `STS2_PPO_FEATURES` gate.
//!
//! A candidate encoder input lives behind a name here so the DEFAULT
//! featurization stays byte-identical to every committed champion while the
//! idea is trained and screened. Python reads the same env var into
//! `schema.ACTIVE_FEATURES` at import time; this side has to agree, because a
//! campaign trains through `ppo/encoder.py` and then *serves* the resulting
//! champion through [`super::encoder`] (`run_arena`, the workbench backend,
//! the advisor exe). A feature that only one of the two honours produces a net
//! evaluated on an observation it never saw in training.
//!
//! Unlike the `emb_boss` / `N_CONT` appends, a feature gated here is NOT
//! self-describing: it changes what goes into an EXISTING slot, so the tensor
//! shapes are identical either way and neither consumer can detect the
//! mismatch from the graph it loaded. The env var is the whole pairing
//! mechanism — `PPOConfig.features` records it into the checkpoint's saved
//! cfg, and `trainer.train` refuses to run when the two disagree.

use std::sync::OnceLock;

/// Every declared feature name. Mirrors `ppo.schema.KNOWN_FEATURES`.
pub const KNOWN: &[&str] = &[Features::REWARD_UPGRADE, Features::CARD_MODS];

/// Which experiment features are on for this featurization.
///
/// `Default` is "none enabled", which is the schema every committed champion
/// was trained at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Features {
    /// Card-reward options carry their real upgrade flag rather than a
    /// hardcoded `false` (GH #545). `roll_for_upgrade` pre-upgrades non-rare
    /// reward options 25% / 50% of the time in acts 2 / 3 below A7 (12.5% /
    /// 25% at A7+), on the most-exercised surface in training — but every
    /// champion to date was trained with the flag dropped, and holds an
    /// unconditional anti-upgraded prior (`wb-27d55ac0`) that would run
    /// BACKWARDS if it were handed the truth mid-life. Hence: off by default,
    /// on for the campaign that retrains with it.
    pub reward_upgrade: bool,
    /// Every card token carries the card's own enchantment (GH #947): its id
    /// in the enchantment column (`cat[11]`) and its amount in
    /// [`super::encoder::C_ENCH_AMT`]. Covers deck tokens, the deck-op
    /// targets (remove / upgrade / transform / enchant / shop-remove / grant)
    /// and relic-stamped reward options. Without it an enchanted card
    /// featurizes as the bare card. Off by default: no champion was trained
    /// on it, and the enchantment table it writes into was only ever trained
    /// on the state token and the enchant-pick action tokens.
    pub card_mods: bool,
}

impl Features {
    /// The `STS2_PPO_FEATURES` token for [`Self::reward_upgrade`].
    pub const REWARD_UPGRADE: &'static str = "reward_upgrade";
    /// The `STS2_PPO_FEATURES` token for [`Self::card_mods`].
    pub const CARD_MODS: &'static str = "card_mods";

    /// Parse a comma-separated `STS2_PPO_FEATURES` spec.
    ///
    /// # Panics
    ///
    /// On an unknown name, exactly like `ppo.schema.parse_features` raises.
    /// Failing loud on a typo is the point: a misspelled feature would
    /// otherwise run as the default arm and be written down as "this idea
    /// does not help".
    #[must_use]
    pub fn parse(spec: &str) -> Self {
        Self::try_parse(spec).unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::parse`] that returns the error instead of panicking.
    ///
    /// For callers that must not unwind: the PyO3 binding raises this as a
    /// `ValueError` (a panic across the FFI boundary is far worse than an
    /// exception, and `catch_unwind` would not even save it under a
    /// `panic = "abort"` profile).
    ///
    /// # Errors
    ///
    /// On an unknown name, with the same message [`Self::parse`] panics with.
    pub fn try_parse(spec: &str) -> Result<Self, String> {
        let mut out = Self::default();
        for name in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            match name {
                Self::REWARD_UPGRADE => out.reward_upgrade = true,
                Self::CARD_MODS => out.card_mods = true,
                _ => {
                    return Err(format!(
                        "unknown STS2_PPO_FEATURES entry {name:?} (known: {KNOWN:?})"
                    ))
                }
            }
        }
        Ok(out)
    }

    /// The process-wide feature set, parsed once from `STS2_PPO_FEATURES`.
    ///
    /// Cached because the collection workers read it per decision and Python
    /// fixes its own copy at import time — a mid-process change would put the
    /// two encoders out of step, which is the one thing this module exists to
    /// prevent.
    #[must_use]
    pub fn active() -> Self {
        static ACTIVE: OnceLock<Features> = OnceLock::new();
        *ACTIVE.get_or_init(|| Self::parse(&std::env::var("STS2_PPO_FEATURES").unwrap_or_default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_every_feature_off() {
        assert_eq!(Features::default(), Features::parse(""));
        assert!(!Features::default().reward_upgrade);
        assert!(!Features::default().card_mods);
    }

    #[test]
    fn parse_reads_a_comma_separated_spec_with_whitespace() {
        assert!(Features::parse(" reward_upgrade , ").reward_upgrade);
        let both = Features::parse("reward_upgrade,card_mods");
        assert!(both.reward_upgrade && both.card_mods);
    }

    #[test]
    #[should_panic(expected = "unknown STS2_PPO_FEATURES entry")]
    fn an_unknown_name_fails_loud_rather_than_screening_as_a_null() {
        let _ = Features::parse("reward_upgrde");
    }

    #[test]
    fn try_parse_reports_an_unknown_name_instead_of_unwinding() {
        let err = Features::try_parse("reward_upgrde").expect_err("typo must not parse");
        assert!(err.contains("reward_upgrde"), "{err}");
        assert!(err.contains(Features::REWARD_UPGRADE), "{err}");
        assert_eq!(
            Features::try_parse("reward_upgrade"),
            Ok(Features::parse("reward_upgrade"))
        );
    }

    #[test]
    fn known_lists_every_name_parse_accepts() {
        for name in KNOWN {
            assert_ne!(Features::parse(name), Features::default(), "{name}");
        }
    }
}
