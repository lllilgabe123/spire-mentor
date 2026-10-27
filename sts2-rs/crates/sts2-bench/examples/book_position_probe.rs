//! `book_position_probe` — realize position-book entries and explain a
//! miss (GH #494 step 2).
//!
//! The regret column PR #695 added to `mcts_book_sweep` says *where* the
//! deployed search loses value on the book: at 4000 sims five positions
//! carry 55% of all regret, each "one line wins, everything else loses".
//! This probe takes those positions apart, one at a time:
//!
//! 1. **Board.** Hand (with costs), piles, potions, relics, powers, every
//!    enemy's HP / block / intent, and the telegraphed damage total.
//! 2. **Oracle table** (from the book) with actions labelled by card.
//! 3. **Shallow search** in the deployed dialect at each requested budget:
//!    per-seed picks, the pooled root table, and `ΔQ = Q_shallow − Q_oracle`
//!    per child — so the read is "does the search over-rate the pick or
//!    under-rate the gold?", not just "it missed".
//! 4. **One-ply static values.** Apply each legal action to a clone and
//!    print the resulting HP / block / energy and the leaf evaluator's
//!    score of that child state. If the static leaf already prefers the
//!    wrong child, the error is value-side at depth 1.
//! 5. **Child searches.** A deeper search from each child state; its root
//!    value is what the search would believe about that child with the
//!    budget concentrated there. A gold child whose deep local value
//!    matches the oracle while the shallow root Q does not is a budget /
//!    exploration problem; a child that stays mispriced under a deep local
//!    search is a value problem.
//! 6. **Principal variation** of the shallow search at the top budget.
//! 7. **Commutation with gold.** For every other action `a`, apply `a`
//!    then gold and gold then `a`; if the two states are identical (piles
//!    compared as sets, draw order kept) the pick is a re-ordering of the
//!    same turn and its true cost is zero whatever the oracle's Q table
//!    says. Otherwise the axis differences are printed.
//! 8. **Forced-first-move playouts.** Force gold, and force each action
//!    the shallow search actually picked, then let the deployed agent play
//!    the fight out `N` times. Win rate and final HP are the regret in
//!    real units — oracle-free, and immune to the mean-backup shading
//!    that makes a rarely-visited oracle child's Q pessimistic.
//!
//! Realization mirrors `mcts_book_sweep::realize_verified` exactly and
//! hash-checks against the frozen `state_hash`.
//!
//! Two book-wide modes reuse the same pieces: `commute-table` prints the
//! re-ordering verdict for every (position, non-gold action) pair, and
//! `playout-regret --rows <book_rows.json>` prices every miss in a sweep
//! dump by forced-first-move playouts of gold versus the pick.
//!
//! ```sh
//! cargo run --release -p sts2-bench --example book_position_probe -- \
//!     <book_json> <pool_root> <book_id>[,<book_id>...] \
//!     [--sims 1000,4000] [--seeds 4] [--child-sims 4000] [--child-seeds 2] \
//!     [--playouts 32] [--playout-sims 1000] [--seed-salt N] [--determinize] [--playout-all]
//! cargo run --release -p sts2-bench --example book_position_probe -- \
//!     <book> <pool> root-table [--motif gold_potion] [--sims 1000,4000] [--seeds 4] \
//!     [--child-sims 4000] [--child-seeds 2]
//! cargo run --release -p sts2-bench --example book_position_probe -- \
//!     <book_json> <pool_root> commute-table
//! cargo run --release -p sts2-bench --example book_position_probe -- \
//!     <book_json> <pool_root> playout-regret --rows <book_rows.json> \
//!     [--playouts 32] [--playout-sims 1000]
//! ```

#[path = "common/mod.rs"]
mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::Deserialize;
use sts2_core::monsters::Intent;
use sts2_core::potions::PotionKind;
use sts2_core::{
    is_terminal, legal_actions, Action, CardUid, CombatPhase, CombatState, StepResult,
};
use sts2_mcts::{HiddenInfoMode, LeafEval, Mcts, MctsConfig, RootOutcome};

#[derive(Debug, Deserialize)]
struct Book {
    #[serde(default = "default_warmup")]
    max_warmup_plies: u32,
    entries: Vec<BookEntry>,
}

const fn default_warmup() -> u32 {
    8
}

#[derive(Debug, Deserialize)]
struct BookEntry {
    book_id: u32,
    character: String,
    tier: String,
    floor: u16,
    fixture_rel_path: String,
    seed: u64,
    state_hash: u64,
    gold_action: String,
    #[serde(default)]
    motifs: Vec<String>,
    #[serde(default)]
    oracle_q: Vec<OracleChild>,
    /// Second copies of the gold card (`combat_position_book hard`).
    #[serde(default)]
    gold_aliases: Vec<String>,
    /// The playout table of a `hard` book entry; `None` on oracle books.
    #[serde(default)]
    playout: Option<EntryPlayout>,
}

#[derive(Debug, Deserialize)]
struct EntryPlayout {
    table: Vec<PlayoutChild>,
}

#[derive(Debug, Deserialize)]
struct PlayoutChild {
    action: String,
    #[serde(default)]
    aliases: Vec<String>,
    n: u32,
    wins: u32,
}

impl BookEntry {
    /// `true` for the gold action or one of its aliases.
    fn is_gold(&self, action_dbg: &str) -> bool {
        self.gold_action == action_dbg || self.gold_aliases.iter().any(|a| a == action_dbg)
    }
    /// Playout win rate of `action_dbg` from a `hard` book's table.
    fn playout_win(&self, action_dbg: &str) -> Option<f64> {
        let t = self.playout.as_ref()?;
        t.table
            .iter()
            .find(|c| c.action == action_dbg || c.aliases.iter().any(|a| a == action_dbg))
            .map(|c| f64::from(c.wins) / f64::from(c.n.max(1)))
    }
}

#[derive(Debug, Deserialize)]
struct OracleChild {
    action: String,
    q: f32,
    stderr: f32,
    visits: u64,
}

/// Deployed advisor dialect — must match `mcts_book_sweep::arm_config` and
/// `combat_position_book::oracle_config` or the probe explains a different
/// search than the one the regret was measured on.
fn deployed_config(sims: u32, seed: u64) -> MctsConfig {
    MctsConfig {
        max_simulations: sims,
        seed,
        max_rollout_depth: 200,
        hidden_info: HiddenInfoMode::BeliefDraw,
        c_puct: 2.5,
        draw_widening_c: 2.0,
        ..MctsConfig::default()
    }
}

/// Same seed schedule as `mcts_book_sweep::trial_seed`, so seed `t` here
/// reproduces the sweep's seed-`t` pick on the same position.
fn trial_seed(t: u32) -> u64 {
    u64::from(t)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x51ED_2701)
}

fn realize(pool_root: &Path, e: &BookEntry, warmup: u32) -> Result<CombatState, String> {
    let path = pool_root.join(&e.fixture_rel_path);
    let mut state = common::realize_fixture(&path, e.seed)?;
    common::warmup_plies(&mut state, e.seed, warmup);
    if state.is_over || state.phase != CombatPhase::PlayerPlayPhase {
        return Err("no decision after warm-up".into());
    }
    let h = sts2_core::state_hash(&state);
    if h != e.state_hash {
        return Err(format!(
            "STATE HASH MISMATCH: realized {h:#x}, book {:#x}",
            e.state_hash
        ));
    }
    Ok(state)
}

// ─── Labels ─────────────────────────────────────────────────────────────────

fn card_at(state: &CombatState, hand_index: u8) -> String {
    let Some(uid) = state.player.hand.get(usize::from(hand_index)).copied() else {
        return format!("h{hand_index}?");
    };
    let Some(card) = state.cards.iter().find(|c| c.uid == uid) else {
        return format!("h{hand_index}?");
    };
    let plus = if card.upgraded { "+" } else { "" };
    format!("{:?}{plus}({})", card.card_id, card.cost)
}

fn enemy_name(state: &CombatState, slot: u8) -> String {
    let Some(uid) = state.enemy_uids.get(usize::from(slot)) else {
        return format!("enemy{slot}?");
    };
    state.creatures.iter().find(|c| c.id == uid.0).map_or_else(
        || format!("enemy{slot}?"),
        |c| {
            c.species
                .map_or_else(|| c.monster_id.0.to_string(), |k| format!("{k:?}"))
        },
    )
}

fn label(state: &CombatState, a: &Action) -> String {
    match a {
        Action::PlayCard { hand_index, target } => {
            let card = card_at(state, *hand_index);
            match target {
                Some(t) => format!("h{hand_index} {card} → {}[{t}]", enemy_name(state, *t)),
                None => format!("h{hand_index} {card}"),
            }
        }
        Action::DrinkPotion {
            potion_slot,
            target,
        } => {
            let name = state
                .player
                .potions
                .get(usize::from(*potion_slot))
                .map_or("potion?", |p| p.python_name());
            match target {
                Some(t) => format!("drink {name} → {}[{t}]", enemy_name(state, *t)),
                None => format!("drink {name}"),
            }
        }
        Action::EndTurn => "EndTurn".to_string(),
        other => format!("{other:?}"),
    }
}

// ─── Board ──────────────────────────────────────────────────────────────────

fn pile_summary(state: &CombatState, uids: &[sts2_core::CardUid]) -> String {
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for uid in uids {
        if let Some(card) = state.cards.iter().find(|c| c.uid == *uid) {
            let plus = if card.upgraded { "+" } else { "" };
            *counts
                .entry(format!("{:?}{plus}", card.card_id))
                .or_default() += 1;
        }
    }
    counts
        .into_iter()
        .map(|(k, n)| if n > 1 { format!("{k}×{n}") } else { k })
        .collect::<Vec<_>>()
        .join(", ")
}

fn powers_of(state: &CombatState, uids: &[sts2_core::PowerUid]) -> String {
    let v: Vec<String> = uids
        .iter()
        .filter_map(|puid| state.powers.iter().find(|p| p.id == puid.0))
        .map(|p| format!("{:?}={}", p.kind, p.amount))
        .collect();
    if v.is_empty() {
        "-".to_string()
    } else {
        v.join(" ")
    }
}

fn intent_str(i: Option<&Intent>) -> String {
    match i {
        None => "-".to_string(),
        Some(Intent::Attack { damage }) => format!("Attack {damage}"),
        Some(Intent::MultiAttack { damage, hits }) => format!("Attack {damage}×{hits}"),
        Some(other) => format!("{other:?}"),
    }
}

fn print_board(state: &CombatState) {
    let p = &state.player;
    println!(
        "  round {}  player HP {}/{} block {} energy {}/{}  powers: {}",
        state.round,
        p.current_hp,
        p.max_hp,
        p.block,
        p.energy,
        p.max_energy,
        powers_of(state, &p.powers)
    );
    let relics: Vec<String> = state
        .relics
        .iter()
        .map(|r| format!("{:?}", r.kind))
        .collect();
    let potions: Vec<&str> = p.potions.iter().map(|k| k.python_name()).collect();
    println!("  relics: {}   potions: {:?}", relics.join(", "), potions);
    println!("  hand:");
    for (i, uid) in p.hand.iter().enumerate() {
        if let Some(card) = state.cards.iter().find(|c| c.uid == *uid) {
            let plus = if card.upgraded { "+" } else { "" };
            println!(
                "    h{i} {:?}{plus}  cost {}  {:?}  dmg {} blk {}",
                card.card_id, card.cost, card.card_type, card.base_damage, card.base_block
            );
        }
    }
    println!(
        "  draw ({}): {}",
        p.draw.len(),
        pile_summary(state, &p.draw)
    );
    println!(
        "  discard ({}): {}",
        p.discard.len(),
        pile_summary(state, &p.discard)
    );
    if !p.exhaust.is_empty() {
        println!(
            "  exhaust ({}): {}",
            p.exhaust.len(),
            pile_summary(state, &p.exhaust)
        );
    }
    let mut incoming = 0_i32;
    for (slot, uid) in state.enemy_uids.iter().enumerate() {
        let Some(c) = state.creatures.iter().find(|c| c.id == uid.0) else {
            continue;
        };
        if c.current_hp <= 0 {
            println!("    enemy[{slot}] {} DEAD", enemy_name(state, slot as u8));
            continue;
        }
        if let Some(i) = c.intent.as_ref() {
            incoming += match *i {
                Intent::Attack { damage } => damage.max(0),
                Intent::MultiAttack { damage, hits } => damage.max(0) * i32::from(hits),
                _ => 0,
            };
        }
        println!(
            "    enemy[{slot}] {}  HP {}/{} block {}  intent {}  powers: {}",
            enemy_name(state, slot as u8),
            c.current_hp,
            c.max_hp,
            c.block,
            intent_str(c.intent.as_ref()),
            powers_of(state, &c.powers)
        );
    }
    println!(
        "  telegraphed damage {incoming} vs HP+block {}",
        p.current_hp + p.block
    );
}

// ─── Pooled root tables ─────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct Acc {
    visits: u64,
    sum: f64,
    sq: f64,
    wins: u64,
    losses: u64,
    ties: u64,
}

impl Acc {
    fn add(&mut self, o: &RootOutcome) {
        self.visits += u64::from(o.visits);
        self.sum += f64::from(o.value_sum);
        self.sq += f64::from(o.value_sq_sum);
        self.wins += u64::from(o.wins);
        self.losses += u64::from(o.losses);
        self.ties += u64::from(o.ties);
    }
    #[allow(clippy::cast_precision_loss)]
    fn q(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.sum / self.visits as f64
        }
    }
    #[allow(clippy::cast_precision_loss)]
    fn se(&self) -> f64 {
        if self.visits == 0 {
            return 0.0;
        }
        let n = self.visits as f64;
        let m = self.sum / n;
        ((self.sq / n - m * m).max(0.0) / n).sqrt()
    }
    #[allow(clippy::cast_precision_loss)]
    fn pct(&self, x: u64) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            100.0 * x as f64 / self.visits as f64
        }
    }
}

fn oracle_q_of(e: &BookEntry, action_dbg: &str) -> Option<f32> {
    e.oracle_q
        .iter()
        .find(|c| c.action == action_dbg)
        .map(|c| c.q)
}

fn print_pooled(state: &CombatState, e: &BookEntry, pooled: &BTreeMap<String, Acc>) {
    let legal = legal_actions(state);
    let by_dbg: BTreeMap<String, &Action> = legal.iter().map(|a| (format!("{a:?}"), a)).collect();
    let mut rows: Vec<(&String, &Acc)> = pooled.iter().collect();
    rows.sort_by(|a, b| {
        b.1.q()
            .partial_cmp(&a.1.q())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!(
        "    {:<44} {:>7} {:>7} {:>7} {:>5} {:>5} {:>5} {:>8} {:>7}",
        "action", "visits", "Q", "se", "win%", "loss%", "tie%", "Q_orac", "ΔQ"
    );
    for (dbg, acc) in rows {
        let lbl = by_dbg
            .get(dbg)
            .map_or_else(|| dbg.clone(), |a| label(state, a));
        let mark = if *dbg == e.gold_action { "*" } else { " " };
        let (oq, dq) = oracle_q_of(e, dbg).map_or_else(
            || ("   n/a".to_string(), "    n/a".to_string()),
            |q| {
                (
                    format!("{q:+8.3}"),
                    format!("{:+7.3}", acc.q() - f64::from(q)),
                )
            },
        );
        println!(
            "   {mark}{:<44} {:>7} {:>+7.3} {:>7.4} {:>5.1} {:>5.1} {:>5.1} {} {}",
            lbl,
            acc.visits,
            acc.q(),
            acc.se(),
            acc.pct(acc.wins),
            acc.pct(acc.losses),
            acc.pct(acc.ties),
            oq,
            dq
        );
    }
}

fn pv_labels(state: &CombatState, pv: &[Action]) -> Vec<String> {
    let mut s = state.clone();
    let mut out = Vec::new();
    for a in pv {
        out.push(label(&s, a));
        if !matches!(
            sts2_core::apply(&mut s, a.clone()),
            StepResult::Advanced | StepResult::PendingChoice
        ) {
            break;
        }
        if s.is_over {
            break;
        }
    }
    out
}

// ─── Per-position probe ─────────────────────────────────────────────────────

struct Opts {
    sims: Vec<u32>,
    seeds: u32,
    child_sims: u32,
    child_seeds: u32,
    playouts: u32,
    playout_sims: u32,
    playout_all: bool,
    /// Mixed into every playout seed. `0` reproduces the seeds a
    /// `combat_position_book hard` build used, so its table can be
    /// re-derived; any other value gives *fresh* fights — the way to price a
    /// playout-set gold without the winner's-curse bias of the playouts that
    /// selected it.
    seed_salt: u64,
    /// Re-deal the hidden information before every playout
    /// (`sts2_core::blindfold::blindfold_root`: unknown draw order and the
    /// hidden RNG streams become a pure function of the playout seed). Off,
    /// every playout of a position runs on the ONE deal frozen in the
    /// fixture, so a "best first move" is the best move *for that deal* —
    /// knowledge no honest search has. On, the playouts average over deals
    /// and price the move the way the player faces it.
    determinize: bool,
}

// ─── Commutation with gold ──────────────────────────────────────────────────

/// Identity of an action that survives another action being applied
/// first: the card by uid (hand indices shift), the potion by kind (slots
/// shift). `None` for actions that have no stable identity across a ply.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ActionKey {
    Card(CardUid, Option<u8>),
    Potion(PotionKind, Option<u8>),
}

fn action_key(state: &CombatState, a: &Action) -> Option<ActionKey> {
    match a {
        Action::PlayCard { hand_index, target } => state
            .player
            .hand
            .get(usize::from(*hand_index))
            .map(|uid| ActionKey::Card(*uid, *target)),
        Action::DrinkPotion {
            potion_slot,
            target,
        } => state
            .player
            .potions
            .get(usize::from(*potion_slot))
            .map(|k| ActionKey::Potion(*k, *target)),
        _ => None,
    }
}

fn resolve_key(state: &CombatState, key: &ActionKey) -> Option<Action> {
    legal_actions(state)
        .into_iter()
        .find(|a| action_key(state, a).as_ref() == Some(key))
}

/// Apply `first` then the action `second_key` names; `None` when either
/// play is rejected, opens a choice, ends the combat, or leaves the play
/// phase — a re-ordering is only a re-ordering if both orders leave the
/// agent standing in the same kind of situation.
fn apply_pair(base: &CombatState, first: &Action, second_key: &ActionKey) -> Option<CombatState> {
    let mut s = base.clone();
    if sts2_core::apply(&mut s, first.clone()) != StepResult::Advanced {
        return None;
    }
    if s.is_over || s.phase != CombatPhase::PlayerPlayPhase {
        return None;
    }
    let second = resolve_key(&s, second_key)?;
    if sts2_core::apply(&mut s, second) != StepResult::Advanced {
        return None;
    }
    if s.is_over || s.phase != CombatPhase::PlayerPlayPhase {
        return None;
    }
    Some(s)
}

/// Piles whose *order* carries no game information are compared as sets.
/// The draw pile is left alone — its order is the future.
fn canonical(mut s: CombatState) -> CombatState {
    s.player.hand.sort_by_key(|u| u.0);
    s.player.discard.sort_by_key(|u| u.0);
    s.player.exhaust.sort_by_key(|u| u.0);
    s
}

enum Commute {
    /// Both orders reach the identical state.
    Same,
    /// Same state except on the dominance axes; `(Δenemy_hp, Δplayer_hp,
    /// Δplayer_block)` of `a`-first minus gold-first, player-favourable
    /// positive.
    AxesOnly(i32, i32, i32),
    /// Something else differs (RNG stream, a power, a pile).
    Differs,
    /// One of the orders is not playable as a pair.
    NotAPair,
}

fn commutes_with_gold(state: &CombatState, a: &Action, gold: &Action) -> Commute {
    let (Some(ka), Some(kg)) = (action_key(state, a), action_key(state, gold)) else {
        return Commute::NotAPair;
    };
    let (Some(ag), Some(ga)) = (apply_pair(state, a, &kg), apply_pair(state, gold, &ka)) else {
        return Commute::NotAPair;
    };
    let (ag, ga) = (canonical(ag), canonical(ga));
    if ag == ga {
        return Commute::Same;
    }
    let (x, y) = (
        common::dominance::axes_of(&ag),
        common::dominance::axes_of(&ga),
    );
    let (mut nag, mut nga) = (ag, ga);
    common::dominance::neutralize_axes(&mut nag);
    common::dominance::neutralize_axes(&mut nga);
    if nag == nga {
        Commute::AxesOnly(
            y.enemy_hp - x.enemy_hp,
            x.player_hp - y.player_hp,
            x.player_block - y.player_block,
        )
    } else {
        Commute::Differs
    }
}

// ─── Forced-first-move playouts ─────────────────────────────────────────────

struct PlayoutOut {
    won: bool,
    final_hp: i32,
    actions: u32,
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// Salt for the per-playout re-deal, so the deal seed and the search seed
/// of one playout are not the same number.
const DETERMINIZE_SALT: u64 = 0xDEA1_0000_0000_0001;

/// Force `first`, then let the deployed agent finish the fight.
fn playout(
    start: &CombatState,
    first: &Action,
    fight_seed: u64,
    sims: u32,
    determinize: bool,
) -> PlayoutOut {
    let mut state = start.clone();
    if determinize {
        sts2_core::blindfold::blindfold_root(&mut state, fight_seed ^ DETERMINIZE_SALT);
    }
    let _ = sts2_core::apply(&mut state, first.clone());
    let mut count: u32 = 0;
    while !is_terminal(&state) && count < 600 {
        if legal_actions(&state).is_empty() {
            let _ = sts2_core::apply(&mut state, Action::EndTurn);
            count += 1;
            continue;
        }
        let step_seed = fight_seed ^ u64::from(count).wrapping_mul(GOLDEN);
        let mut m = Mcts::new(state.clone(), deployed_config(sims, step_seed));
        let a = m.search_n(sims);
        let _ = sts2_core::apply(&mut state, a);
        count += 1;
    }
    let final_hp = state.player.current_hp.max(0);
    PlayoutOut {
        won: is_terminal(&state) && state.player_won,
        final_hp,
        actions: count,
    }
}

#[allow(clippy::cast_precision_loss)]
fn print_playouts(state: &CombatState, gold: &Action, actions: &[Action], opts: &Opts) {
    let n = opts.playouts;
    println!(
        "\n  forced-first-move playouts: deployed agent @ {} sims finishes the fight, {n} fights per action (paired seeds)",
        opts.playout_sims
    );
    let tasks: Vec<(usize, u32)> = (0..actions.len())
        .flat_map(|i| (0..n).map(move |t| (i, t)))
        .collect();
    let outs: Vec<(usize, u32, PlayoutOut)> = tasks
        .into_par_iter()
        .map(|(i, t)| {
            let seed = 0xF1_6A7_u64
                .wrapping_add(u64::from(t).wrapping_mul(GOLDEN))
                .wrapping_add(sts2_core::state_hash(state))
                ^ opts.seed_salt.wrapping_mul(GOLDEN);
            (
                i,
                t,
                playout(
                    state,
                    &actions[i],
                    seed,
                    opts.playout_sims,
                    opts.determinize,
                ),
            )
        })
        .collect();
    let mut per: Vec<Vec<Option<&PlayoutOut>>> = vec![vec![None; n as usize]; actions.len()];
    for (i, t, o) in &outs {
        per[*i][*t as usize] = Some(o);
    }
    let gold_i = actions.iter().position(|a| a == gold);
    println!(
        "    {:<44} {:>5} {:>6} {:>7} {:>7} {:>8} {:>12}",
        "action", "wins", "win%", "±se", "HP_end", "actions", "vs gold W/L"
    );
    for (i, a) in actions.iter().enumerate() {
        let rows: Vec<&PlayoutOut> = per[i].iter().flatten().copied().collect();
        let wins = rows.iter().filter(|o| o.won).count();
        let p = wins as f64 / rows.len() as f64;
        let se = (p * (1.0 - p) / rows.len() as f64).sqrt();
        let hp = rows.iter().map(|o| f64::from(o.final_hp)).sum::<f64>() / rows.len() as f64;
        let acts = rows.iter().map(|o| f64::from(o.actions)).sum::<f64>() / rows.len() as f64;
        let vs = gold_i.map_or_else(String::new, |g| {
            if g == i {
                "(gold)".to_string()
            } else {
                let mut w = 0;
                let mut l = 0;
                for (x, y) in per[i].iter().zip(per[g].iter()) {
                    if let (Some(x), Some(y)) = (x, y) {
                        if x.won && !y.won {
                            w += 1;
                        }
                        if !x.won && y.won {
                            l += 1;
                        }
                    }
                }
                format!("{w}/{l}")
            }
        });
        let mark = if a == gold { "*" } else { " " };
        println!(
            "   {mark}{:<44} {:>5} {:>6.1} {:>7.1} {:>7.1} {:>8.1} {:>12}",
            label(state, a),
            wins,
            100.0 * p,
            100.0 * se,
            hp,
            acts,
            vs
        );
    }
}

fn probe(state: &CombatState, e: &BookEntry, opts: &Opts) {
    let t0 = Instant::now();
    println!(
        "\n══════ #{} {} {} floor {}  {}  motifs {:?}",
        e.book_id, e.character, e.tier, e.floor, e.fixture_rel_path, e.motifs
    );
    print_board(state);

    let legal = legal_actions(state);
    let gold = legal.iter().find(|a| format!("{a:?}") == e.gold_action);
    println!(
        "\n  GOLD: {}",
        gold.map_or_else(|| e.gold_action.clone(), |a| label(state, a))
    );

    println!("\n  oracle table (book, pooled 2 methods × 3 seeds):");
    let by_dbg: BTreeMap<String, &Action> = legal.iter().map(|a| (format!("{a:?}"), a)).collect();
    for c in &e.oracle_q {
        let lbl = by_dbg
            .get(&c.action)
            .map_or_else(|| c.action.clone(), |a| label(state, a));
        let mark = if c.action == e.gold_action { "*" } else { " " };
        println!(
            "   {mark}{:<44} q={:+.3} se={:.4} visits={}",
            lbl, c.q, c.stderr, c.visits
        );
    }

    // ── shallow searches ──
    let mut top_pv: Vec<Action> = Vec::new();
    let mut mispicks: Vec<Action> = Vec::new();
    for &sims in &opts.sims {
        let mut pooled: BTreeMap<String, Acc> = BTreeMap::new();
        let mut picks: Vec<String> = Vec::new();
        for t in 0..opts.seeds {
            let mut m = Mcts::new(state.clone(), deployed_config(sims, trial_seed(t)));
            let a = m.search_n(sims);
            let dbg = format!("{a:?}");
            if dbg != e.gold_action && !mispicks.contains(&a) {
                mispicks.push(a.clone());
            }
            picks.push(format!(
                "{}{}",
                if dbg == e.gold_action { "✓ " } else { "✗ " },
                label(state, &a)
            ));
            for o in m.root_outcomes() {
                pooled.entry(format!("{:?}", o.action)).or_default().add(&o);
            }
            if t == 0 {
                top_pv = m.get_pv();
            }
        }
        println!("\n  deployed search @ {sims} sims × {} seeds:", opts.seeds);
        for (t, p) in picks.iter().enumerate() {
            println!("    seed {t}: {p}");
        }
        print_pooled(state, e, &pooled);
        println!(
            "    PV (seed 0): {}",
            pv_labels(state, &top_pv).join("  →  ")
        );
    }

    // ── one-ply static + child searches ──
    let leaf = LeafEval::default();
    println!(
        "\n  one ply deeper: static leaf of the child state, and a {}-sim × {} local search from it",
        opts.child_sims, opts.child_seeds
    );
    println!(
        "    {:<44} {:>6} {:>7} {:>9} {:>8} {:>8} {:>8} {:>8}",
        "action", "step", "leaf", "hp/blk/en", "enemyHP", "childQ", "childQ*", "Q_orac"
    );
    for a in &legal {
        let mut child = state.clone();
        let step = sts2_core::apply(&mut child, a.clone());
        let dbg = format!("{a:?}");
        let mark = if dbg == e.gold_action { "*" } else { " " };
        let enemy_hp: i32 = state
            .enemy_uids
            .iter()
            .filter_map(|u| child.creatures.iter().find(|c| c.id == u.0))
            .map(|c| c.current_hp.max(0))
            .sum();
        let leaf_v = leaf.evaluate(&child);
        let child_q =
            if child.is_over || child.pending_choice.is_some() || step == StepResult::PendingChoice
            {
                None
            } else if child.phase != CombatPhase::PlayerPlayPhase {
                // EndTurn: the enemy turn has resolved and we are in the next
                // player turn (or the loop is mid-way). Search from wherever
                // the engine left the state if a decision exists.
                if legal_actions(&child).is_empty() {
                    None
                } else {
                    Some(child_search(&child, opts))
                }
            } else {
                Some(child_search(&child, opts))
            };
        let oq = oracle_q_of(e, &dbg);
        let fmt = |v: Option<f64>| v.map_or_else(|| "  n/a".to_string(), |q| format!("{q:+8.3}"));
        println!(
            "   {mark}{:<44} {:>6} {:>+7.3} {:>3}/{:<2}/{:<2} {:>8} {:>8} {:>8} {:>8}",
            label(state, a),
            format!("{step:?}").chars().take(6).collect::<String>(),
            leaf_v,
            child.player.current_hp,
            child.player.block,
            child.player.energy,
            enemy_hp,
            fmt(child_q.map(|c| c.0)),
            fmt(child_q.map(|c| c.1)),
            fmt(oq.map(f64::from)),
        );
    }
    println!(
        "    childQ = visit-weighted mean of the local search (what a parent backs up); \
         childQ* = its best well-visited grandchild (what the line is worth)."
    );

    // ── commutation ──
    if let Some(gold) = gold {
        println!("\n  does the other action commute with gold? (a then gold  vs  gold then a)");
        for a in &legal {
            if a == gold {
                continue;
            }
            let verdict = match commutes_with_gold(state, a, gold) {
                Commute::Same => "SAME STATE — a re-ordering; true cost 0".to_string(),
                Commute::AxesOnly(de, dh, db) => format!(
                    "same except axes: Δenemy_hp {de:+} Δplayer_hp {dh:+} Δblock {db:+} (a-first minus gold-first, player-favourable positive)"
                ),
                Commute::Differs => "differs (RNG, power or pile)".to_string(),
                Commute::NotAPair => "not playable as a pair".to_string(),
            };
            let picked = if mispicks.contains(a) { "PICKED " } else { "" };
            println!("    {picked}{:<44} {verdict}", label(state, a));
        }
    }

    // ── playouts ──
    if opts.playouts > 0 {
        if let Some(gold) = gold {
            let mut actions: Vec<Action> = vec![gold.clone()];
            if opts.playout_all {
                actions.extend(legal.iter().filter(|a| *a != gold).cloned());
            } else {
                actions.extend(mispicks.iter().cloned());
            }
            print_playouts(state, gold, &actions, opts);
        }
    }
    println!("  [{:.1}s]", t0.elapsed().as_secs_f64());
}

/// Two values of a deeper search from `child`, both in root-Q units:
/// the visit-weighted root mean (what a parent node would back up — the
/// quantity the oracle table stores per child), and the best well-visited
/// grandchild's mean (what the line is actually worth once the search has
/// found it). The gap between the two is the mean-backup shading.
fn child_search(child: &CombatState, opts: &Opts) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut n = 0_u64;
    let mut best_sum = 0.0_f64;
    for t in 0..opts.child_seeds {
        let mut m = Mcts::new(
            child.clone(),
            deployed_config(opts.child_sims, trial_seed(100 + t)),
        );
        let _ = m.search_n(opts.child_sims);
        let outs = m.root_outcomes();
        let total: u64 = outs.iter().map(|o| u64::from(o.visits)).sum();
        for o in &outs {
            sum += f64::from(o.value_sum);
            n += u64::from(o.visits);
        }
        // "Well-visited" = at least 5% of the search, so a lucky 3-rollout
        // child cannot pose as the best line.
        let floor = total / 20;
        best_sum += outs
            .iter()
            .filter(|o| u64::from(o.visits) >= floor.max(1))
            .map(|o| f64::from(o.mean_value))
            .fold(f64::NEG_INFINITY, f64::max);
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = if n == 0 { 0.0 } else { sum / n as f64 };
    (mean, best_sum / f64::from(opts.child_seeds))
}

/// Book-wide commutation table: one TSV line per (entry, non-gold legal
/// action) — `COMMUTE <book_id> <action_dbg> <verdict> <Δenemy_hp> <Δhp>
/// <Δblock>` — for joining against a sweep dump's misses.
fn commute_table(pool_root: &Path, book: &Book) {
    println!("COMMUTE\tbook_id\taction\tverdict\td_enemy_hp\td_player_hp\td_block");
    let rows: Vec<String> = book
        .entries
        .par_iter()
        .flat_map_iter(|e| {
            let mut out = Vec::new();
            let Ok(state) = realize(pool_root, e, book.max_warmup_plies) else {
                return out;
            };
            let legal = legal_actions(&state);
            let Some(gold) = legal.iter().find(|a| format!("{a:?}") == e.gold_action) else {
                return out;
            };
            for a in &legal {
                if a == gold {
                    continue;
                }
                let (v, de, dh, db) = match commutes_with_gold(&state, a, gold) {
                    Commute::Same => ("same", 0, 0, 0),
                    Commute::AxesOnly(de, dh, db) => ("axes", de, dh, db),
                    Commute::Differs => ("differs", 0, 0, 0),
                    Commute::NotAPair => ("not_pair", 0, 0, 0),
                };
                out.push(format!(
                    "COMMUTE\t{}\t{:?}\t{v}\t{de}\t{dh}\t{db}",
                    e.book_id, a
                ));
            }
            out
        })
        .collect();
    for r in rows {
        println!("{r}");
    }
}

/// One `mcts_book_sweep --dump` row; only the join keys are read.
#[derive(Debug, Deserialize)]
struct DumpRow {
    book_id: u32,
    pick: String,
    agree: bool,
}

/// Book-wide playout regret: for every position the dump missed, force
/// gold and force each distinct mis-pick, then let the deployed agent
/// finish the fight `n` times with paired seeds. One TSV line per
/// (position, action) — `PLAYOUT <book_id> <character> <tier> <is_gold>
/// <action_dbg> <n> <wins> <mean_final_hp> <mean_actions>` — plus the
/// per-seed win vectors so a consumer can pair them.
fn playout_regret(pool_root: &Path, book: &Book, rows_path: &Path, opts: &Opts) {
    let rows: Vec<DumpRow> =
        serde_json::from_str(&std::fs::read_to_string(rows_path).expect("read dump"))
            .expect("parse dump");
    let mut picks_by_id: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for r in rows.iter().filter(|r| !r.agree) {
        let v = picks_by_id.entry(r.book_id).or_default();
        if !v.contains(&r.pick) {
            v.push(r.pick.clone());
        }
    }
    eprintln!(
        "[playout-regret] {} missed positions, {} distinct (position, pick) pairs, {} playouts each",
        picks_by_id.len(),
        picks_by_id.values().map(Vec::len).sum::<usize>(),
        opts.playouts
    );
    println!("PLAYOUT\tbook_id\tcharacter\ttier\tis_gold\taction\tn\twins\tmean_final_hp\tmean_actions\twins_by_seed");
    // (entry, action_dbg) tasks; gold first so a partial log is usable.
    let mut tasks: Vec<(&BookEntry, String)> = Vec::new();
    for (id, picks) in &picks_by_id {
        let Some(e) = book.entries.iter().find(|e| e.book_id == *id) else {
            continue;
        };
        tasks.push((e, e.gold_action.clone()));
        for p in picks {
            tasks.push((e, p.clone()));
        }
    }
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = tasks.len();
    let t0 = Instant::now();
    tasks.par_iter().for_each(|(e, action_dbg)| {
        let Ok(state) = realize(pool_root, e, book.max_warmup_plies) else {
            eprintln!("[playout-regret] #{} did not realize", e.book_id);
            return;
        };
        let Some(action) = legal_actions(&state)
            .into_iter()
            .find(|a| format!("{a:?}") == *action_dbg)
        else {
            eprintln!(
                "[playout-regret] #{} action {action_dbg} not legal",
                e.book_id
            );
            return;
        };
        let outs: Vec<PlayoutOut> = (0..opts.playouts)
            .into_par_iter()
            .map(|t| {
                let seed = 0xF1_6A7_u64
                    .wrapping_add(u64::from(t).wrapping_mul(GOLDEN))
                    .wrapping_add(sts2_core::state_hash(&state))
                    ^ opts.seed_salt.wrapping_mul(GOLDEN);
                playout(&state, &action, seed, opts.playout_sims, opts.determinize)
            })
            .collect();
        #[allow(clippy::cast_precision_loss)]
        let n = outs.len() as f64;
        let wins = outs.iter().filter(|o| o.won).count();
        let hp = outs.iter().map(|o| f64::from(o.final_hp)).sum::<f64>() / n;
        let acts = outs.iter().map(|o| f64::from(o.actions)).sum::<f64>() / n;
        let bits: String = outs.iter().map(|o| if o.won { '1' } else { '0' }).collect();
        println!(
            "PLAYOUT\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.1}\t{}",
            e.book_id,
            e.character,
            e.tier,
            u8::from(*action_dbg == e.gold_action),
            action_dbg,
            outs.len(),
            wins,
            hp,
            acts,
            bits
        );
        let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if k.is_multiple_of(20) || k == total {
            eprintln!(
                "[playout-regret] {k}/{total} tasks  {:.0}s",
                t0.elapsed().as_secs_f64()
            );
        }
    });
}

/// Book-wide diagnostic of *how* the deployed search misses (GH #446 hard
/// book): for every entry (optionally only those carrying `--motif <tag>`),
/// at each `--sims` and `--seeds` seed, the deployed search's root table —
/// each child's visits and mean Q, with the gold (or alias) and the picked
/// child flagged — as `ROOT` rows; and for the gold child and every
/// distinct picked child, the static leaf of the child state plus a
/// `--child-sims × --child-seeds` local search from it, as `CHILD` rows.
/// A `hard` book's playout win rate is joined onto both so the search's
/// own estimate can be read against the realized value.
///
/// Reading: a gold child whose Q is inside the tie band but has fewer
/// visits than the pick lost on visits; a gold child whose Q is below the
/// band is under-valued by the search, and the `CHILD` rows say whether a
/// deeper search from that child agrees (the evaluation cannot see it) or
/// not (the root under-estimates it).
fn root_table(pool_root: &Path, book: &Book, motif: Option<&str>, opts: &Opts) {
    let entries: Vec<&BookEntry> = book
        .entries
        .iter()
        .filter(|e| motif.is_none_or(|m| e.motifs.iter().any(|t| t == m)))
        .collect();
    eprintln!(
        "[root-table] {} entries{}; sims {:?} x {} seeds; child {} sims x {} seeds",
        entries.len(),
        motif.map_or(String::new(), |m| format!(" with motif {m}")),
        opts.sims,
        opts.seeds,
        opts.child_sims,
        opts.child_seeds
    );
    println!("ROOT\tbook_id\ttier\tsims\tseed\taction\tvisits\tq\tis_gold\tis_pick\tplayout_win");
    println!("CHILD\tbook_id\ttier\taction\tis_gold\tleaf\tchild_mean\tchild_best\tplayout_win");
    let done = std::sync::atomic::AtomicUsize::new(0);
    let t0 = Instant::now();
    let blocks: Vec<String> = entries
        .par_iter()
        .map(|e| {
            let mut out = String::new();
            let state = match realize(pool_root, e, book.max_warmup_plies) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("[root-table] #{}: {err}", e.book_id);
                    return out;
                }
            };
            let pw = |dbg: &str| {
                e.playout_win(dbg)
                    .map_or("-".to_string(), |w| format!("{w:.3}"))
            };
            let mut picked: Vec<String> = Vec::new();
            for &sims in &opts.sims {
                for t in 0..opts.seeds {
                    let mut m = Mcts::new(state.clone(), deployed_config(sims, trial_seed(t)));
                    let pick = format!("{:?}", m.search_n(sims));
                    if !picked.contains(&pick) {
                        picked.push(pick.clone());
                    }
                    for o in m.root_outcomes() {
                        let dbg = format!("{:?}", o.action);
                        out.push_str(&format!(
                            "ROOT\t{}\t{}\t{sims}\t{t}\t{dbg}\t{}\t{:.4}\t{}\t{}\t{}\n",
                            e.book_id,
                            e.tier,
                            o.visits,
                            o.mean_value,
                            u8::from(e.is_gold(&dbg)),
                            u8::from(dbg == pick),
                            pw(&dbg)
                        ));
                    }
                }
            }
            // Children worth a deeper look: gold, and everything the search picked.
            let legal = legal_actions(&state);
            let mut targets: Vec<&Action> = legal
                .iter()
                .filter(|a| e.is_gold(&format!("{a:?}")))
                .take(1)
                .collect();
            for p in &picked {
                if let Some(a) = legal.iter().find(|a| format!("{a:?}") == *p) {
                    if !targets.iter().any(|t| format!("{t:?}") == *p) {
                        targets.push(a);
                    }
                }
            }
            let leaf = LeafEval::default();
            for a in targets {
                let dbg = format!("{a:?}");
                let mut child = state.clone();
                let step = sts2_core::apply(&mut child, a.clone());
                let leaf_v = leaf.evaluate(&child);
                let searched = if child.is_over
                    || child.pending_choice.is_some()
                    || step == StepResult::PendingChoice
                    || legal_actions(&child).is_empty()
                {
                    None
                } else {
                    Some(child_search(&child, opts))
                };
                out.push_str(&format!(
                    "CHILD\t{}\t{}\t{dbg}\t{}\t{leaf_v:.4}\t{}\t{}\t{}\n",
                    e.book_id,
                    e.tier,
                    u8::from(e.is_gold(&dbg)),
                    searched.map_or("-".to_string(), |(m, _)| format!("{m:.4}")),
                    searched.map_or("-".to_string(), |(_, b)| format!("{b:.4}")),
                    pw(&dbg)
                ));
            }
            let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if k.is_multiple_of(10) || k == entries.len() {
                eprintln!(
                    "[root-table] {k}/{} entries  {:.0}s",
                    entries.len(),
                    t0.elapsed().as_secs_f64()
                );
            }
            out
        })
        .collect();
    for b in blocks {
        print!("{b}");
    }
}

fn parse_list(s: &str) -> Vec<u32> {
    s.split(',')
        .filter(|x| !x.is_empty())
        .map(|x| x.parse().expect("integer list"))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!(
            "usage: book_position_probe <book_json> <pool_root> <id>[,<id>...] \
             [--sims 1000,4000] [--seeds 4] [--child-sims 4000] [--child-seeds 2]"
        );
        std::process::exit(2);
    }
    let book_path = PathBuf::from(&args[0]);
    let pool_root = PathBuf::from(&args[1]);
    let mut opts = Opts {
        sims: vec![1000, 4000],
        seeds: 4,
        child_sims: 4000,
        child_seeds: 2,
        playouts: 0,
        playout_sims: 1000,
        seed_salt: 0,
        playout_all: false,
        determinize: false,
    };
    let mut rows_path: Option<PathBuf> = None;
    let mut motif: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        if args[i] == "--playout-all" {
            opts.playout_all = true;
            i += 1;
            continue;
        }
        if args[i] == "--determinize" {
            opts.determinize = true;
            i += 1;
            continue;
        }
        let v = args.get(i + 1).expect("flag value");
        match args[i].as_str() {
            "--sims" => opts.sims = parse_list(v),
            "--seeds" => opts.seeds = v.parse().expect("--seeds"),
            "--child-sims" => opts.child_sims = v.parse().expect("--child-sims"),
            "--child-seeds" => opts.child_seeds = v.parse().expect("--child-seeds"),
            "--playouts" => opts.playouts = v.parse().expect("--playouts"),
            "--playout-sims" => opts.playout_sims = v.parse().expect("--playout-sims"),
            "--seed-salt" => opts.seed_salt = v.parse().expect("--seed-salt"),
            "--rows" => rows_path = Some(PathBuf::from(v)),
            "--motif" => motif = Some(v.clone()),
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }

    let book: Book = serde_json::from_str(&std::fs::read_to_string(&book_path).expect("read book"))
        .expect("parse book");
    match args[2].as_str() {
        "commute-table" => {
            commute_table(&pool_root, &book);
            return;
        }
        "root-table" => {
            root_table(&pool_root, &book, motif.as_deref(), &opts);
            return;
        }
        "playout-regret" => {
            let rows = rows_path.expect("playout-regret needs --rows <book_rows.json>");
            if opts.playouts == 0 {
                opts.playouts = 32;
            }
            playout_regret(&pool_root, &book, &rows, &opts);
            return;
        }
        _ => {}
    }
    let ids = parse_list(&args[2]);
    println!(
        "book {} ({} entries, warm-up {} plies); pool {}",
        book_path.display(),
        book.entries.len(),
        book.max_warmup_plies,
        pool_root.display()
    );
    for id in ids {
        let Some(e) = book.entries.iter().find(|e| e.book_id == id) else {
            eprintln!("no entry #{id}");
            continue;
        };
        match realize(&pool_root, e, book.max_warmup_plies) {
            Ok(state) => probe(&state, e, &opts),
            Err(err) => eprintln!("#{id}: {err}"),
        }
    }
}
