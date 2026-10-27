//! `net_expand_order_cost` — what does `MctsConfig::net_expand_order`
//! (GH #830 rung 4) cost per simulation? (perf probe, not a gate)
//!
//! Runs the deployed BeliefDraw dialect (`apply_mode_dialect`: c_puct 2.5 at
//! 1000 sims, draw widening 2.0, EndTurn widening 5.787 / 0.436, the built-in
//! Ironclad playout net) with the knob off and on, on the same states and
//! search seeds, single thread. States: a 25-card random-pool Ironclad deck
//! against each encounter in `ENCOUNTERS`, advanced `--plies` random plies
//! into the fight so the root is not always the opening hand. The two arms
//! alternate which runs first per state, so host drift hits both alike.
//!
//! ```text
//! cargo build --release --example net_expand_order_cost
//! target/release/examples/net_expand_order_cost --decks 6 --sims 1000
//! ```
//!
//! Prints one line per state, then the totals: summed wall time per arm,
//! `ratio = on / off` of the summed per-simulation time, the median of the
//! per-state ratios, the reorder counters and how often the root pick
//! changed.

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use sts2_content::cards::spec_index::IRONCLAD_SPECS;
use sts2_content::cards::CardSpec;
use sts2_core::{CharacterId, CombatState};
use sts2_mcts::{ExpandOrderStats, HiddenInfoMode, Mcts, MctsConfig};

const DECK_SIZE: usize = 25;
const ENCOUNTERS: &[&str] = &[
    "MAWLER_NORMAL",
    "SLIMES_NORMAL",
    "BYGONE_EFFIGY_ELITE",
    "PHROG_PARASITE_ELITE",
    "VANTOM_BOSS",
    "THE_KIN_BOSS",
];

fn random_deck(seed: u64, size: usize) -> Vec<CardSpec> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let mut rng = sts2_core::dotnet_rng::Rng::new((seed ^ 0xDEC0_DEC0_DEC0_DEC0) as i32);
    (0..size)
        .map(|_| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            #[allow(clippy::cast_sign_loss)]
            let idx = rng.next_int_exclusive(0, IRONCLAD_SPECS.len() as i32) as usize;
            IRONCLAD_SPECS[idx]()
        })
        .collect()
}

/// One timed search: `(seconds, chosen action, reorder counters)`.
fn timed(state: &CombatState, cfg: MctsConfig) -> (f64, String, ExpandOrderStats) {
    let start = Instant::now();
    let mut m = Mcts::new(state.clone(), cfg);
    let action = m.search();
    let secs = start.elapsed().as_secs_f64();
    (secs, format!("{action:?}"), m.expand_order_stats())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        f64::midpoint(v[n / 2 - 1], v[n / 2])
    }
}

fn main() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();

    let mut n_decks = 6_u32;
    let mut sims = 1000_u32;
    let mut plies = 3_u32;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().expect("flag needs a value");
        match a.as_str() {
            "--decks" => n_decks = val().parse().expect("u32"),
            "--sims" => sims = val().parse().expect("u32"),
            "--plies" => plies = val().parse().expect("u32"),
            other => panic!("unknown flag {other}"),
        }
    }
    assert!(
        std::env::var_os("STS2_ROLLOUT_NET").is_none(),
        "unset STS2_ROLLOUT_NET: this probe measures the built-in Ironclad net"
    );

    let cfg_for = |seed: u64, on: bool| {
        let mut c = MctsConfig {
            max_simulations: sims,
            max_rollout_depth: 200,
            seed,
            hidden_info: HiddenInfoMode::BeliefDraw,
            ..MctsConfig::default()
        };
        c.apply_mode_dialect();
        c.net_expand_order = on;
        c
    };
    let state_for = |enc: &str, deck_idx: u32| {
        let deck = random_deck(
            u64::from(deck_idx).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            DECK_SIZE,
        );
        let engine_seed = u64::from(deck_idx) * 1000 + 1;
        let mut s = common::build_ironclad_state_with_deck(enc, engine_seed, &deck);
        s.player.character = Some(CharacterId::Ironclad);
        common::warmup_plies(&mut s, engine_seed ^ 0xA11, plies);
        s
    };

    // Warmup (net parse, allocator, lazy tables; untimed).
    {
        let s = state_for(ENCOUNTERS[0], 999);
        let _ = timed(&s, cfg_for(0xFEED, false));
        let _ = timed(&s, cfg_for(0xFEED, true));
    }

    let (mut off_total, mut on_total) = (0.0_f64, 0.0_f64);
    let mut ratios = Vec::new();
    let mut flips = 0_u32;
    let mut n = 0_u32;
    let mut sum = ExpandOrderStats::default();
    for (e, enc) in ENCOUNTERS.iter().enumerate() {
        for deck_idx in 0..n_decks {
            let s = state_for(enc, deck_idx);
            if sts2_core::is_terminal(&s) || sts2_core::legal_actions(&s).len() < 2 {
                continue;
            }
            let seed = ((u64::from(deck_idx) << 8) | u64::try_from(e).unwrap_or(0)) ^ 0x5EED_5EED;
            let on_first = (n % 2) == 1;
            let (off, on) = if on_first {
                let on = timed(&s, cfg_for(seed, true));
                (timed(&s, cfg_for(seed, false)), on)
            } else {
                let off = timed(&s, cfg_for(seed, false));
                (off, timed(&s, cfg_for(seed, true)))
            };
            off_total += off.0;
            on_total += on.0;
            ratios.push(on.0 / off.0);
            flips += u32::from(off.1 != on.1);
            n += 1;
            sum.considered += on.2.considered;
            sum.scored += on.2.scored;
            sum.reordered += on.2.reordered;
            sum.first_changed += on.2.first_changed;
            println!(
                "{enc:<22} deck={deck_idx} off={:.1}ms on={:.1}ms ratio={:.3} flip={} \
                 considered={} scored={} reordered={} first_changed={}",
                off.0 * 1e3,
                on.0 * 1e3,
                on.0 / off.0,
                off.1 != on.1,
                on.2.considered,
                on.2.scored,
                on.2.reordered,
                on.2.first_changed,
            );
        }
    }
    let per_sim = |t: f64| t / f64::from(n) / f64::from(sims) * 1e6;
    println!(
        "TOTAL states={n} sims={sims} off={off_total:.2}s on={on_total:.2}s \
         per_sim_off={:.1}us per_sim_on={:.1}us ratio={:.3} median_ratio={:.3} \
         root_flips={flips}/{n} considered={} scored={} reordered={} first_changed={}",
        per_sim(off_total),
        per_sim(on_total),
        on_total / off_total,
        median(ratios),
        sum.considered,
        sum.scored,
        sum.reordered,
        sum.first_changed,
    );
}
