//! Potion opportunity-cost table for the root-level drink charge
//! (GH #447 follow-up).
//!
//! `STS2_RUNAI_POTION_OC=<path.json>` loads a per-(potion, act) table of
//! carry-forward values in **HP-equivalents**; each combat search converts
//! the held belt into per-slot root-Q charges
//! ([`sts2_core::victory_hp_equiv_to_q`]) and hands them to
//! [`sts2_mcts::MctsConfig::potion_charge_q`]. The table format is the
//! shared artifact between the critic-counterfactual probe (option A) and
//! the randomized-drink causal fit (option B):
//!
//! ```text
//! {
//!   "version": 1,
//!   "character": "ironclad",
//!   "units": "hp_equiv",
//!   "source": "jitter_fit",
//!   "default_oc": 8.0,
//!   "entries": [ {"potion": "POTION.FIRE_POTION", "act": 1, "oc": 4.0} ]
//! }
//! ```
//!
//! Failure policy: this seam gates measurement campaigns, so like
//! `STS2_RUNAI_HONEST` / `STS2_RUNAI_POTION_WEIGHT` it PANICS on any
//! invalid input instead of silently falling back — a typo must not
//! poison an A/B arm. That includes the λ-pairing rule: the charge prices
//! the potion at the root, so the LEAF potion term must be explicitly
//! zeroed (`STS2_RUNAI_POTION_WEIGHT=0`) or the value would be counted
//! twice.

use std::collections::BTreeMap;
#[cfg(feature = "ppo")]
use std::sync::OnceLock;

#[cfg(feature = "ppo")]
use sts2_core::potions::CharacterId;
use sts2_core::potions::PotionKind;
use sts2_core::run_manager::RunManager;
use sts2_core::runstate::RoomType;
use sts2_core::{victory_hp_equiv_to_q, CombatState};

#[cfg(feature = "ppo")]
use crate::ppo::encoder::featurize_with;
#[cfg(feature = "ppo")]
use crate::ppo::features::Features;
#[cfg(feature = "ppo")]
use crate::ppo::net::{features_for, onnx_model_path_for, PpoNet};
#[cfg(feature = "ppo")]
use crate::ppo::obs::{ActionDesc, PpoObs};
#[cfg(feature = "ppo")]
use crate::ppo::policy::build_obs;

/// Final act of a run (table acts are {1,2,3}); a boss fight in this
/// act has no future to carry potions into, so critic mode charges
/// zero there — the same rule the table encodes via `acts_present`.
#[cfg(feature = "ppo")]
const FINAL_ACT: u8 = 3;

/// Loaded OC table: `(potion token, act)` → HP-equivalents, with a
/// `default_oc` fallback for pairs the fit had no data on.
#[derive(Debug, Clone)]
pub(crate) struct PotionOcTable {
    default_oc: f32,
    entries: BTreeMap<(String, u8), f32>,
    /// Acts that appear in ANY entry — the final act's boss is detected
    /// as "the next act has no entries at all".
    acts_present: std::collections::BTreeSet<u8>,
}

impl PotionOcTable {
    /// OC in HP-equivalents for one potion token (`"POTION.X"`) in `act`.
    fn oc_hp_equiv(&self, token: &str, act: u8) -> f32 {
        self.entries
            .get(&(token.to_string(), act))
            .copied()
            .unwrap_or(self.default_oc)
    }

    /// Per-slot root-Q charges for the current combat belt. Slots past
    /// the belt length stay 0 (no root child references them).
    ///
    /// `is_boss` matters: during a boss fight the belt's carry value is
    /// its value in the NEXT act (you only keep what you carry through),
    /// and at the FINAL act's boss — no next act in the table — the
    /// charge is zero for every slot. The 2026-08-19 mech-panel autopsy
    /// found 4/11 charged-arm losses dying at bosses HOLDING 1–2 potions
    /// (baseline: 0/12) because the flat per-act price kept taxing
    /// drinks in the one fight with no future to save for.
    pub(crate) fn charge_slots(&self, belt: &[PotionKind], act: u8, is_boss: bool) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        let effective_act = if is_boss { act.saturating_add(1) } else { act };
        if is_boss && !self.acts_present.contains(&effective_act) {
            return out; // final boss: carrying past it is worthless
        }
        for (slot, kind) in belt.iter().enumerate().take(out.len()) {
            let token = format!("POTION.{}", kind.python_name());
            out[slot] = victory_hp_equiv_to_q(self.oc_hp_equiv(&token, effective_act));
        }
        out
    }

    /// Full-belt swap decision (GH #252 surface, GH #447 acquisition
    /// follow-up): `Some(slot)` = discard that held slot and take
    /// `offered`, `None` = decline. Rule: replace the WORST held potion
    /// (argmin OC, first-min on ties) iff the offer prices STRICTLY
    /// higher — so free tokens (OC 0) are the first to go and an offer
    /// no better than the current minimum is left on the reward screen.
    pub(crate) fn resolve_swap(
        &self,
        belt: &[PotionKind],
        offered: PotionKind,
        act: u8,
    ) -> Option<usize> {
        let oc = |kind: PotionKind| {
            let token = format!("POTION.{}", kind.python_name());
            self.oc_hp_equiv(&token, act)
        };
        let (min_slot, min_oc) = belt
            .iter()
            .enumerate()
            .map(|(i, &k)| (i, oc(k)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).expect("OC values are finite"))?;
        (oc(offered) > min_oc).then_some(min_slot)
    }
}

/// The resolved charge source: a static `(potion, act)` table or the
/// live critic counterfactual (`STS2_RUNAI_POTION_OC=critic`, plan
/// 2026-08-20-ppo-potion-bootstrap P1). Both feed the same two consumers
/// — the per-slot root-Q charge and the full-belt swap rule.
#[derive(Debug)]
pub(crate) enum PotionOc {
    Table(PotionOcTable),
    /// Live critic counterfactual — needs the ONNX runtime (`ppo` feature).
    #[cfg(feature = "ppo")]
    Critic(CriticOc),
}

/// Per-fight-class multiplier on the root charge
/// (`STS2_RUNAI_POTION_OC_CLASS_SCALE="<trash>,<elite>,<boss>"`,
/// default `1,1,1` = exactly today's behavior, bit-identical because
/// multiplying an `f32` by 1.0 is exact).
///
/// Why: the charge prices a potion's FUTURE value and leaves "is this
/// fight worth a potion?" to the search — which is only half-working.
/// Measured 2026-08-23 on the deployed stack vs the expert human corpus
/// (drinks per fight):
///
/// ```text
///            trash  elite  boss      shape N:E:B
///   ours      0.43   0.63  1.18      1 : 1.4 : 2.7
///   experts   0.175  0.455 0.895     1 : 2.6 : 5.1
/// ```
///
/// Elites and bosses are close to right in absolute terms; essentially
/// the whole residual error is trash over-drinking (2.5× the expert
/// rate). A class multiplier lets the charge be steep exactly where the
/// error is instead of uniformly.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClassScale {
    trash: f32,
    elite: f32,
    boss: f32,
}

impl Default for ClassScale {
    fn default() -> Self {
        Self {
            trash: 1.0,
            elite: 1.0,
            boss: 1.0,
        }
    }
}

impl ClassScale {
    /// Multiplier for the room the fight is in. Anything that is not an
    /// elite or boss room (including `Unknown`) counts as trash — the
    /// conservative direction, since trash is where the charge is meant
    /// to bite.
    pub(crate) fn for_room(self, room: Option<RoomType>) -> f32 {
        match room {
            Some(RoomType::Boss) => self.boss,
            Some(RoomType::Elite) => self.elite,
            _ => self.trash,
        }
    }

    /// Pure resolver: `"<trash>,<elite>,<boss>"`, each finite and >= 0.
    /// Unset ⇒ all-ones. Malformed ⇒ panic (measurement seam).
    pub(crate) fn resolve(env: Option<&str>) -> Self {
        let Some(raw) = env else {
            return Self::default();
        };
        let vals: Vec<f32> = raw
            .split(',')
            .map(|t| {
                t.trim()
                    .parse::<f32>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .unwrap_or_else(|| {
                        panic!(
                            "STS2_RUNAI_POTION_OC_CLASS_SCALE: invalid {raw:?} \
                             (three finite >= 0 values: trash,elite,boss)"
                        )
                    })
            })
            .collect();
        match vals.as_slice() {
            [t, e, b] => Self {
                trash: *t,
                elite: *e,
                boss: *b,
            },
            _ => panic!(
                "STS2_RUNAI_POTION_OC_CLASS_SCALE: {raw:?} must have exactly 3 \
                 comma-separated values (trash,elite,boss)"
            ),
        }
    }
}

/// Whether a fight's belt is priced as carried into the NEXT act — the
/// `is_boss` argument of [`PotionOcTable::charge_slots`] and its critic
/// twin. True for a boss fight, EXCEPT the first boss of an A10
/// DoubleBoss act: there the belt carries into the second boss of the
/// SAME act, so it keeps its current-act price. Before this rule the
/// first act-3 boss at A10 was read as "the final boss" and charged zero
/// — the opposite of the carry it actually has (GH #830 elite-gap: the
/// second boss beats 74% of the A10 runs that reach it).
pub(crate) fn prices_next_act(is_boss: bool, awaiting_second_boss: bool) -> bool {
    is_boss && !awaiting_second_boss
}

impl PotionOc {
    /// Per-slot root-Q charges for the current combat belt (dispatching
    /// twin of [`PotionOcTable::charge_slots`]).
    pub(crate) fn charge_slots_ctx(
        &mut self,
        rm: &RunManager,
        combat: &CombatState,
        is_boss: bool,
        class_scale: ClassScale,
    ) -> [f32; 8] {
        let is_boss = prices_next_act(is_boss, rm.run_state.awaiting_second_boss());
        let mut charges = match self {
            PotionOc::Table(t) => {
                t.charge_slots(&combat.player.potions, rm.run_state.act_index, is_boss)
            }
            #[cfg(feature = "ppo")]
            PotionOc::Critic(c) => c.charge_slots(rm, combat, is_boss),
        };
        // Exact-1.0 short-circuit: the default must be BIT-identical to
        // the unscaled path, and `x * 1.0f32` already is — the compare
        // only skips the loop, so an epsilon band would be wrong here
        // (a 0.999 scale is a real setting and must apply).
        #[allow(clippy::float_cmp)]
        let scaled = class_scale.for_room(rm.run_state.current_room.map(|r| r.room_type)) != 1.0;
        if scaled {
            let k = class_scale.for_room(rm.run_state.current_room.map(|r| r.room_type));
            for q in &mut charges {
                *q *= k;
            }
        }
        charges
    }

    /// Full-belt swap decision (dispatching twin of
    /// [`PotionOcTable::resolve_swap`]).
    pub(crate) fn resolve_swap_ctx(
        &mut self,
        rm: &RunManager,
        offered: PotionKind,
    ) -> Option<usize> {
        match self {
            PotionOc::Table(t) => {
                t.resolve_swap(&rm.run_state.potions, offered, rm.run_state.act_index)
            }
            #[cfg(feature = "ppo")]
            PotionOc::Critic(c) => c.resolve_swap(rm, offered),
        }
    }

    /// Test constructor — critic mode without going through the env
    /// resolver.
    #[cfg(all(test, feature = "ppo"))]
    pub(crate) fn critic_for_tests(scale: f32) -> Self {
        // The frozen deckmut_it150 critic's measured rates
        // (probe_critic.json `hp_per_value_point_by_act`).
        PotionOc::Critic(CriticOc::new(scale, [0.000_72, 0.000_88, 0.000_5]))
    }
}

/// Live critic-counterfactual charge source: prices each held potion as
/// `scale × max(V(s) − V(s∖potion), 0)`, converted to HP-equivalents by
/// the critic's own HP slope (`dV_hp10`, the probe's hp+10 trick from
/// `sts2-pycli/python/diag_potion_oc_probe.py`) and then to root-Q via
/// [`victory_hp_equiv_to_q`]. Unlike the static table this sees deck,
/// belt size, HP, map position and (for boss-conditioned champions) the
/// upcoming boss.
///
/// The net is the FROZEN champion resolved by
/// [`onnx_model_path_for`] — during training `STS2_RUNAI_PPO` points it
/// at the previous champion (the target-network rule: a learning critic
/// must never price its own environment). Loaded lazily on first use,
/// once per policy instance (one instance per run).
#[cfg(feature = "ppo")]
pub(crate) struct CriticOc {
    scale: f32,
    /// V-units per HP, per act (index = act−1, clamped) — the FIXED
    /// exchange rate that turns a potion's dV into HP-equivalents.
    /// Measured once per frozen critic (median of the probe's hp+10
    /// slopes; `hp_per_value_point_by_act` in
    /// `logs/2026-08-19-potion-oc/probe_critic.json`) and passed via
    /// `STS2_RUNAI_POTION_OC_HP_RATE`. A PER-STATE slope was tried
    /// first and rejected: a single finite difference of the net is
    /// noise of the same magnitude as the potion dVs themselves (the
    /// P1 fixture found V(80/80) < V(70/80)), and it sits in the
    /// denominator — the fixed rate keeps the context-awareness in the
    /// numerator where the signal is.
    v_per_hp_by_act: [f32; 3],
    net: OnceLock<PpoNet>,
    /// Per-fight memo: the charge is a function of (floor, belt), and
    /// each of the ~80 combat decisions per run rebuilds its config —
    /// without this the run would pay ~80× the ~4 net evals per fight.
    cache: Option<CriticCache>,
}

#[cfg(feature = "ppo")]
struct CriticCache {
    floor: i32,
    belt: Vec<PotionKind>,
    charges: [f32; 8],
}

// The cache is a transient per-fight memo — eliding it is the point.
#[cfg(feature = "ppo")]
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for CriticOc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CriticOc")
            .field("scale", &self.scale)
            .field("v_per_hp_by_act", &self.v_per_hp_by_act)
            .field("net_loaded", &self.net.get().is_some())
            .finish()
    }
}

#[cfg(feature = "ppo")]
impl CriticOc {
    fn new(scale: f32, v_per_hp_by_act: [f32; 3]) -> Self {
        Self {
            scale,
            v_per_hp_by_act,
            net: OnceLock::new(),
            cache: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    /// The frozen charge critic. `STS2_RUNAI_POTION_OC_NET` pins it to an
    /// explicit ONNX path — REQUIRED whenever `STS2_RUNAI_PPO` selects a
    /// per-arm policy net (a gate sweep), because `onnx_model_path_for`
    /// honors that override and would silently price each arm with its
    /// own critic, breaking the shared-frozen-critic design the
    /// pre-registration commits to. Unset, the committed champion
    /// resolves as before (training collection, deployed default).
    fn net(&self, character: CharacterId) -> &PpoNet {
        self.net.get_or_init(|| {
            let path = if let Ok(p) = std::env::var("STS2_RUNAI_POTION_OC_NET") {
                let p = std::path::PathBuf::from(p);
                assert!(
                    p.is_file(),
                    "STS2_RUNAI_POTION_OC_NET: {} is not a file",
                    p.display()
                );
                p
            } else {
                assert!(
                    std::env::var("STS2_RUNAI_PPO").is_err(),
                    "STS2_RUNAI_POTION_OC=critic with STS2_RUNAI_PPO set: the \
                     charge critic would follow the per-arm policy override — \
                     pin the frozen critic explicitly with STS2_RUNAI_POTION_OC_NET"
                );
                onnx_model_path_for(character).unwrap_or_else(|| {
                    panic!(
                        "STS2_RUNAI_POTION_OC=critic: no champion ONNX for {character:?} \
                         (set STS2_RUNAI_POTION_OC_NET to point at the frozen critic)"
                    )
                })
            };
            PpoNet::load(&path).unwrap_or_else(|e| {
                panic!("STS2_RUNAI_POTION_OC=critic: load {}: {e}", path.display())
            })
        })
    }

    /// V(run state) with the belt overridden to `belt` — the combat
    /// belt for charges, the run belt for swaps; overriding both from
    /// ONE source means a mid-fight drink can never desync obs from the
    /// slots being priced. `featurize` ignores `obs.phase`; the state
    /// token's surface comes from the action list, so the single `Move`
    /// frames every eval as a `"path"` decision — the most common meta
    /// surface, in-distribution for the critic — and cancels out of
    /// every difference below because it is identical on both sides.
    /// The encoder schema the critic net was trained at.
    ///
    /// Mirrors [`Self::net`]'s override precedence exactly: an explicitly
    /// pinned critic (`STS2_RUNAI_POTION_OC_NET`) is a net no manifest entry
    /// describes, so `STS2_PPO_FEATURES` declares its schema; otherwise the
    /// committed champion resolves at the spec `champions.toml` records for
    /// it. Getting this wrong is silent — a gated feature rewrites an existing
    /// slot rather than adding a column, so the tensor looks identical either
    /// way and only the prices come out wrong.
    fn features(character: CharacterId) -> Features {
        if std::env::var_os("STS2_RUNAI_POTION_OC_NET").is_some() {
            return Features::active();
        }
        features_for(character)
    }

    fn eval_v(net: &PpoNet, obs: &PpoObs, features: Features) -> f32 {
        let actions = [ActionDesc::Move { col: 0 }];
        let feat = featurize_with(obs, &actions, features);
        match net.infer(&feat) {
            Ok((_, v)) => v,
            Err(e) => panic!("STS2_RUNAI_POTION_OC=critic: net inference failed: {e}"),
        }
    }

    /// `(V(s), V(s∖potion) per slot)` for `belt`.
    fn counterfactuals(&self, rm: &RunManager, belt: &[PotionKind]) -> (f32, Vec<f32>) {
        let character = rm.run_state.character_id;
        let net = self.net(character);
        let features = Self::features(character);
        let mut obs = build_obs(rm, "potion_oc", None);
        obs.potions = belt.iter().map(|k| format!("{k:?}")).collect();
        obs.num_potions = belt.len() as i64;
        let v = Self::eval_v(net, &obs, features);
        let v_cf = (0..belt.len())
            .map(|slot| {
                let mut o = obs.clone();
                o.potions.remove(slot);
                o.num_potions -= 1;
                Self::eval_v(net, &o, features)
            })
            .collect();
        (v, v_cf)
    }

    /// The act's fixed V→HP exchange rate (act is 1-based; clamped so
    /// an out-of-range act can never index past the table).
    fn v_per_hp(&self, act: u8) -> f32 {
        self.v_per_hp_by_act[usize::from(act.clamp(1, 3) - 1)]
    }

    /// Critic-mode twin of [`PotionOcTable::charge_slots`]. Boss rules:
    /// final-act boss ⇒ zero (no future to carry into); a NON-final boss
    /// keeps the current-state value unchanged — the table prices act+1
    /// there, but re-framing the obs into the next act would shift every
    /// correlated feature (act_floor, map, …), so the plan accepts the
    /// current-act value as the documented approximation.
    fn charge_slots(&mut self, rm: &RunManager, combat: &CombatState, is_boss: bool) -> [f32; 8] {
        let belt = &combat.player.potions;
        if belt.is_empty() || (is_boss && rm.run_state.act_index >= FINAL_ACT) {
            return [0.0; 8];
        }
        let floor = rm.run_state.total_floor;
        if let Some(c) = &self.cache {
            if c.floor == floor && c.belt.as_slice() == &belt[..] {
                return c.charges;
            }
        }
        let (v, v_cf) = self.counterfactuals(rm, belt);
        let charges =
            critic_charges_from_values(v, self.v_per_hp(rm.run_state.act_index), &v_cf, self.scale);
        self.cache = Some(CriticCache {
            floor,
            belt: belt.to_vec(),
            charges,
        });
        charges
    }

    /// Critic-mode twin of [`PotionOcTable::resolve_swap`]: same
    /// argmin-eviction / strictly-higher-offer rule, on raw critic dVs
    /// (no HP conversion needed — only the comparison matters). The
    /// offer is priced in the evicted slot's place, so "accept" reduces
    /// to `V(swapped belt) > V(current belt)`.
    fn resolve_swap(&mut self, rm: &RunManager, offered: PotionKind) -> Option<usize> {
        let belt: Vec<PotionKind> = rm.run_state.potions.clone();
        if belt.is_empty() {
            return None;
        }
        let (v, v_cf) = self.counterfactuals(rm, &belt);
        let dv_held: Vec<f32> = v_cf.iter().map(|cf| v - cf).collect();
        let min_slot = critic_swap_from_values(&dv_held, f32::INFINITY)?;
        let character = rm.run_state.character_id;
        let net = self.net(character);
        let mut obs = build_obs(rm, "potion_oc", None);
        obs.potions = belt.iter().map(|k| format!("{k:?}")).collect();
        obs.potions[min_slot] = format!("{offered:?}");
        obs.num_potions = belt.len() as i64;
        let dv_offer = Self::eval_v(net, &obs, Self::features(character)) - v_cf[min_slot];
        critic_swap_from_values(&dv_held, dv_offer)
    }
}

/// Pure charge math (unit-tested without a net): per-slot
/// `victory_hp_equiv_to_q(scale × max(v − v_cf[slot], 0) / v_per_hp)`.
/// A non-positive rate cannot express HP-equivalents — charge nothing
/// rather than divide by ~0 (the resolver already rejects such rates;
/// this guard keeps the pure fn total).
#[cfg(feature = "ppo")]
fn critic_charges_from_values(v: f32, v_per_hp: f32, v_cf: &[f32], scale: f32) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    if v_per_hp <= 1e-9 {
        return out;
    }
    for (slot, cf) in v_cf.iter().enumerate().take(out.len()) {
        let hp_equiv = (v - cf).max(0.0) / v_per_hp;
        out[slot] = victory_hp_equiv_to_q(scale * hp_equiv);
    }
    out
}

/// Pure swap rule on critic dVs: evict the argmin-dV held slot
/// (first-min on ties) iff the offer's dV is STRICTLY higher — the
/// table rule's exact shape, so the two modes only differ in where the
/// values come from.
#[cfg(feature = "ppo")]
fn critic_swap_from_values(dv_held: &[f32], dv_offer: f32) -> Option<usize> {
    let (min_slot, min_dv) = dv_held
        .iter()
        .copied()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(&b.1).expect("critic dVs are finite"))?;
    (dv_offer > min_dv).then_some(min_slot)
}

#[derive(serde::Deserialize)]
struct OcFile {
    version: u32,
    #[allow(dead_code)]
    character: String,
    units: String,
    #[allow(dead_code)]
    source: String,
    default_oc: f32,
    entries: Vec<OcEntry>,
}

#[derive(serde::Deserialize)]
struct OcEntry {
    potion: String,
    act: u8,
    oc: f32,
}

/// Parse + validate an OC table. `label` names the source in panics.
pub(crate) fn parse_potion_oc(json: &str, label: &str) -> PotionOcTable {
    let file: OcFile = serde_json::from_str(json)
        .unwrap_or_else(|e| panic!("STS2_RUNAI_POTION_OC: {label} is not a valid OC table: {e}"));
    assert!(
        file.version == 1,
        "STS2_RUNAI_POTION_OC: {label} has version {} (expected 1)",
        file.version
    );
    assert!(
        file.units == "hp_equiv",
        "STS2_RUNAI_POTION_OC: {label} has units {:?} (expected \"hp_equiv\")",
        file.units
    );
    let check = |v: f32, what: &str| {
        assert!(
            v.is_finite() && v >= 0.0,
            "STS2_RUNAI_POTION_OC: {label}: {what} = {v} must be finite and >= 0 \
             (a negative OC is a drink *bonus* — almost certainly a fit sign bug)"
        );
    };
    check(file.default_oc, "default_oc");
    let mut entries = BTreeMap::new();
    for e in &file.entries {
        check(e.oc, &format!("entries[{} act {}].oc", e.potion, e.act));
        assert!(
            entries.insert((e.potion.clone(), e.act), e.oc).is_none(),
            "STS2_RUNAI_POTION_OC: {label}: duplicate entry for ({}, act {})",
            e.potion,
            e.act
        );
    }
    let acts_present = entries.keys().map(|(_, a)| *a).collect();
    PotionOcTable {
        default_oc: file.default_oc,
        entries,
        acts_present,
    }
}

/// Pure resolver (env values injected for testability, mirroring
/// `resolve_tie_eps`): `oc_env` is `STS2_RUNAI_POTION_OC`, `lambda_env`
/// is `STS2_RUNAI_POTION_WEIGHT`. Unset OC ⇒ `None` ⇒ deployed behavior
/// untouched. Set OC requires λ explicitly `0` (leaf term zeroed) and a
/// readable, valid table — anything else panics.
pub(crate) fn resolve_potion_oc(
    oc_env: Option<&str>,
    lambda_env: Option<&str>,
    scale_env: Option<&str>,
    hp_rate_env: Option<&str>,
) -> Option<PotionOc> {
    let mode = oc_env?;
    let lambda_ok = lambda_env
        .and_then(|s| s.trim().parse::<f32>().ok())
        .is_some_and(|v| v == 0.0);
    assert!(
        lambda_ok,
        "STS2_RUNAI_POTION_OC requires STS2_RUNAI_POTION_WEIGHT=0 (got {lambda_env:?}) — \
         the leaf potion term would double-count the value the root charge prices"
    );
    if mode == "critic" {
        return Some(resolve_critic_oc(scale_env, hp_rate_env));
    }
    assert!(
        hp_rate_env.is_none(),
        "STS2_RUNAI_POTION_OC_HP_RATE only applies to STS2_RUNAI_POTION_OC=critic \
         (table mode got {hp_rate_env:?})"
    );
    // The scale knob only exists in critic mode; a stray setting must
    // not silently do nothing in a sweep arm. Checked before the file
    // read so the panic names the real mistake.
    assert!(
        scale_env.is_none(),
        "STS2_RUNAI_POTION_OC_SCALE only applies to STS2_RUNAI_POTION_OC=critic \
         (table mode got {scale_env:?})"
    );
    let json = std::fs::read_to_string(mode)
        .unwrap_or_else(|e| panic!("STS2_RUNAI_POTION_OC: cannot read {mode:?}: {e}"));
    Some(PotionOc::Table(parse_potion_oc(&json, mode)))
}

/// The `STS2_RUNAI_POTION_OC=critic` arm of [`resolve_potion_oc`]:
/// `scale_env` is `STS2_RUNAI_POTION_OC_SCALE`, `hp_rate_env` is
/// `STS2_RUNAI_POTION_OC_HP_RATE`. Split out only so the critic (which
/// needs the ONNX runtime) can be compiled out with the `ppo` feature.
#[cfg(feature = "ppo")]
fn resolve_critic_oc(scale_env: Option<&str>, hp_rate_env: Option<&str>) -> PotionOc {
    let scale = scale_env.map_or(1.5, |s| {
        s.trim()
            .parse::<f32>()
            .ok()
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or_else(|| {
                panic!("STS2_RUNAI_POTION_OC_SCALE: invalid {s:?} (finite > 0 required)")
            })
    });
    let raw = hp_rate_env.unwrap_or_else(|| {
        panic!(
            "STS2_RUNAI_POTION_OC=critic requires STS2_RUNAI_POTION_OC_HP_RATE \
             (V-units per HP: one value, or act1,act2,act3) — measure it from the \
             FROZEN critic's probe (hp_per_value_point_by_act in probe_critic.json); \
             a built-in default would go silently stale at the next promotion"
        )
    });
    let vals: Vec<f32> = raw
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f32>()
                .ok()
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or_else(|| {
                    panic!(
                        "STS2_RUNAI_POTION_OC_HP_RATE: invalid {raw:?} \
                         (finite > 0 values required)"
                    )
                })
        })
        .collect();
    let rates = match vals.as_slice() {
        [one] => [*one; 3],
        [a, b, c] => [*a, *b, *c],
        _ => panic!(
            "STS2_RUNAI_POTION_OC_HP_RATE: {raw:?} must have 1 or 3 \
             comma-separated values (got {})",
            vals.len()
        ),
    };
    PotionOc::Critic(CriticOc::new(scale, rates))
}

/// Without the `ppo` feature there is no ONNX runtime to price the belt
/// with, so critic mode is a loud configuration error (measurement seam,
/// like every other malformed `STS2_RUNAI_POTION_OC*` setting) rather
/// than a silent fall-through to table mode.
#[cfg(not(feature = "ppo"))]
fn resolve_critic_oc(_scale_env: Option<&str>, _hp_rate_env: Option<&str>) -> PotionOc {
    panic!(
        "STS2_RUNAI_POTION_OC=critic requires the `ppo` cargo feature of sts2-runai \
         (tract-onnx inference is compiled out of this build)"
    )
}

/// Randomized per-fight drink masking (`STS2_RUNAI_DRINK_JITTER`) — the
/// data-generation seam for causal opportunity-cost estimation: when the
/// per-fight coin fires, the search runs with
/// [`sts2_mcts::MctsConfig::root_drink_mask`] set, so the fight cannot
/// spend potions and the belt carried OUT of it is exogenous.
///
/// **The coin formula is an interface, not an implementation detail** —
/// the offline fit (`sts2-pycli/python/fit_potion_oc.py`) recomputes it
/// per (run seed, floor) to label which fights were masked without any
/// recording change:
///
/// ```text
/// material = (run_seed as u32 as u64) XOR (total_floor as u32 as u64) XOR (salt << 32)
/// coin     = splitmix64(material)            # 0x9E3779B97F4A7C15 add,
///                                            # 0xBF58476D1CE4E5B9 / 0x94D049BB133111EB mixes
/// masked   = (coin >> 11) / 2^53 < p
/// ```
#[derive(Debug, Clone, Copy)]
pub(crate) struct DrinkJitter {
    p: f64,
    salt: u64,
}

impl DrinkJitter {
    /// Deterministic per-fight coin: does this fight run drink-masked?
    pub(crate) fn masks(&self, run_seed: i32, total_floor: i32) -> bool {
        let material =
            u64::from(run_seed as u32) ^ u64::from(total_floor as u32) ^ (self.salt << 32);
        let coin = super::splitmix64(material);
        #[allow(clippy::cast_precision_loss)] // 53-bit mantissa by construction
        let u = (coin >> 11) as f64 / (1u64 << 53) as f64;
        u < self.p
    }
}

/// Pure `STS2_RUNAI_DRINK_JITTER` resolver: `"<p>:<salt>"` (e.g.
/// `"0.5:3"`), `p ∈ [0, 1]`, salt a u64. Unset ⇒ `None` ⇒ no-op.
/// Malformed ⇒ panic (this seam only exists to generate measurement
/// corpora; a typo must not silently produce an unjittered arm).
pub(crate) fn resolve_drink_jitter(env: Option<&str>) -> Option<DrinkJitter> {
    let raw = env?;
    let parsed = raw.split_once(':').and_then(|(p, salt)| {
        let p = p.trim().parse::<f64>().ok()?;
        let salt = salt.trim().parse::<u64>().ok()?;
        (p.is_finite() && (0.0..=1.0).contains(&p)).then_some(DrinkJitter { p, salt })
    });
    Some(parsed.unwrap_or_else(|| {
        panic!(
            "STS2_RUNAI_DRINK_JITTER={raw:?} is not \"<p>:<salt>\" with p in [0,1] \
             (e.g. \"0.5:3\")"
        )
    }))
}

/// Pure `STS2_RUNAI_POTION_SWAP` resolver: `"1"` enables the GH #252
/// interactive potion swap in the arena's `RunManager` (full-belt potion
/// rewards queue a discard-to-claim decision instead of silently
/// dropping). Unset / `"0"` ⇒ off (historical silent drop); anything else
/// panics (measurement seam, no silent fallback).
///
/// It used to REQUIRE `STS2_RUNAI_POTION_OC`, on the grounds that "the swap
/// rule prices decisions through the OC table, and an unpriced swap surface
/// would resolve arbitrarily". GH #379 removed that hazard: `potion_swap`
/// now has a real encoder surface, so an OC-less swap is priced by the run
/// policy like any other decision rather than resolving arbitrarily.
/// `oc_env` therefore no longer gates the FLAG — it selects WHO DECIDES, via
/// [`swap_is_rule_side`].
pub(crate) fn resolve_potion_swap_flag(swap_env: Option<&str>, _oc_env: Option<&str>) -> bool {
    match swap_env {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => panic!("STS2_RUNAI_POTION_SWAP={other:?} must be unset, \"0\" or \"1\""),
    }
}

/// Who resolves a surfaced full-belt swap: the OC value rule (`true`) or the
/// run policy through its `potion_swap` encoder surface (`false`).
///
/// An OC table means you are running the #479 opportunity-cost measurement,
/// whose arm is defined by the rule deciding; keep that reproducible. With no
/// table the decision is an ordinary encoded one and the net picks — the arm
/// the next campaign trains on (GH #545).
///
/// **Both engines must agree.** `policy/mod.rs` gates its `PotionSwap`
/// interception on this, and `ppo/env.py` gates the mirror-image branch in
/// `_advance` on the same value (read through the PyO3
/// `potion_swap_is_rule_side` binding), or G1 diverges.
#[must_use]
pub(crate) const fn swap_is_rule_side(oc_env: Option<&str>) -> bool {
    oc_env.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{
        "version": 1, "character": "ironclad", "units": "hp_equiv",
        "source": "manual", "default_oc": 8.0,
        "entries": [
            {"potion": "POTION.FIRE_POTION", "act": 1, "oc": 4.0},
            {"potion": "POTION.POWER_POTION", "act": 3, "oc": 16.0},
            {"potion": "POTION.POTION_SHAPED_ROCK", "act": 1, "oc": 0.0}
        ]
    }"#;

    #[test]
    #[allow(clippy::float_cmp)] // exact-by-construction: parsed literals.
    fn lookup_hits_entries_and_falls_back_to_default() {
        let t = parse_potion_oc(GOOD, "test");
        assert!((t.oc_hp_equiv("POTION.FIRE_POTION", 1) - 4.0).abs() < 1e-6);
        assert!((t.oc_hp_equiv("POTION.POWER_POTION", 3) - 16.0).abs() < 1e-6);
        // act miss and potion miss both fall back to default_oc
        assert!((t.oc_hp_equiv("POTION.FIRE_POTION", 2) - 8.0).abs() < 1e-6);
        assert!((t.oc_hp_equiv("POTION.BLOCK_POTION", 1) - 8.0).abs() < 1e-6);
        // an explicit 0 entry (free relic token) is honored, not defaulted
        assert_eq!(t.oc_hp_equiv("POTION.POTION_SHAPED_ROCK", 1), 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // 0.0 entries convert to exactly 0.0.
    fn charge_slots_converts_hp_equiv_to_q_per_slot() {
        let t = parse_potion_oc(GOOD, "test");
        let belt = [PotionKind::FirePotion, PotionKind::PotionShapedRock];
        let slots = t.charge_slots(&belt, 1, false);
        assert!((slots[0] - victory_hp_equiv_to_q(4.0)).abs() < 1e-7);
        assert_eq!(slots[1], 0.0);
        // slots past the belt stay 0
        assert!(slots[2..].iter().all(|&c| c == 0.0));
    }

    /// During a BOSS fight the carry value is the value of holding into
    /// the NEXT act — a boss in act 2 prices Power Potion at its act-3
    /// entry (16), not act 2's default. The 2026-08-19 mech-panel autopsy
    /// motivated this: 4/11 ON-arm losses died at bosses HOLDING potions
    /// (off arm: 0/12) because the flat per-act charge kept taxing drinks
    /// in the one fight where carrying has no future.
    #[test]
    fn boss_fight_prices_carry_into_the_next_act() {
        let t = parse_potion_oc(GOOD, "test");
        let belt = [PotionKind::PowerPotion];
        let slots = t.charge_slots(&belt, 2, true);
        assert!((slots[0] - victory_hp_equiv_to_q(16.0)).abs() < 1e-7);
    }

    /// At the FINAL act's boss there is no next act to carry into: the
    /// charge must be exactly zero for every slot, whatever the table
    /// says per-potion — dying with a drinkable belt is the one mistake
    /// no expert makes.
    #[test]
    #[allow(clippy::float_cmp)]
    fn final_boss_charges_nothing() {
        let t = parse_potion_oc(GOOD, "test");
        let belt = [PotionKind::FirePotion, PotionKind::PowerPotion];
        // GOOD's highest act is 3 => a boss in act 3 has no act-4 carry.
        let slots = t.charge_slots(&belt, 3, true);
        assert!(slots.iter().all(|&c| c == 0.0), "{slots:?}");
    }

    /// The FIRST boss of an A10 DoubleBoss act is not a final boss: the
    /// belt carries into the second boss of the same act, so it is priced
    /// at the current act's value — non-zero — while the second (last)
    /// boss still charges nothing.
    #[test]
    #[allow(clippy::float_cmp)]
    fn first_of_two_bosses_prices_the_current_act() {
        let t = parse_potion_oc(GOOD, "test");
        let belt = [PotionKind::PowerPotion];
        let first = t.charge_slots(&belt, 3, prices_next_act(true, true));
        assert!((first[0] - victory_hp_equiv_to_q(16.0)).abs() < 1e-7);
        let last = t.charge_slots(&belt, 3, prices_next_act(true, false));
        assert_eq!(last[0], 0.0);
        // A non-boss fight is unaffected by the double-boss flag.
        assert!(!prices_next_act(false, true));
        assert!(!prices_next_act(false, false));
    }

    #[test]
    fn unset_env_resolves_to_none() {
        assert!(resolve_potion_oc(None, None, None, None).is_none());
        // λ set without OC is also fine — the sweep seam works alone.
        assert!(resolve_potion_oc(None, Some("12"), None, None).is_none());
    }

    #[test]
    #[should_panic(expected = "requires STS2_RUNAI_POTION_WEIGHT=0")]
    fn oc_without_lambda_zero_panics() {
        resolve_potion_oc(Some("/nonexistent.json"), None, None, None);
    }

    #[test]
    #[should_panic(expected = "requires STS2_RUNAI_POTION_WEIGHT=0")]
    fn oc_with_default_lambda_panics() {
        resolve_potion_oc(Some("/nonexistent.json"), Some("8"), None, None);
    }

    #[test]
    #[should_panic(expected = "cannot read")]
    fn unreadable_path_panics() {
        resolve_potion_oc(Some("/nonexistent-dir/oc.json"), Some("0"), None, None);
    }

    #[test]
    #[should_panic(expected = "not a valid OC table")]
    fn invalid_json_panics() {
        parse_potion_oc("{ nope", "test");
    }

    #[test]
    #[should_panic(expected = "version 2 (expected 1)")]
    fn wrong_version_panics() {
        parse_potion_oc(
            r#"{"version":2,"character":"ironclad","units":"hp_equiv",
                "source":"manual","default_oc":8.0,"entries":[]}"#,
            "test",
        );
    }

    #[test]
    #[should_panic(expected = "must be finite and >= 0")]
    fn negative_oc_panics() {
        parse_potion_oc(
            r#"{"version":1,"character":"ironclad","units":"hp_equiv",
                "source":"manual","default_oc":8.0,
                "entries":[{"potion":"POTION.FIRE_POTION","act":1,"oc":-2.0}]}"#,
            "test",
        );
    }

    #[test]
    fn jitter_coin_is_deterministic_and_rate_accurate() {
        let j = resolve_drink_jitter(Some("0.5:3")).unwrap();
        // Determinism: same (seed, floor) always lands the same side.
        for floor in 0..50 {
            assert_eq!(j.masks(123_456_789, floor), j.masks(123_456_789, floor));
        }
        // Salt changes the pattern (not necessarily every floor).
        let j2 = resolve_drink_jitter(Some("0.5:4")).unwrap();
        let differs = (0..200).any(|f| j.masks(42, f) != j2.masks(42, f));
        assert!(differs, "different salts must produce different coins");
        // Rate over 10k (seed, floor) cells within ±2pp of p.
        let fired = (0..200_i32)
            .flat_map(|s| (0..50_i32).map(move |f| (s, f)))
            .filter(|&(s, f)| j.masks(s.wrapping_mul(7_919), f))
            .count();
        let rate = fired as f64 / 10_000.0;
        assert!(
            (rate - 0.5).abs() < 0.02,
            "rate {rate} not within ±2pp of 0.5"
        );
    }

    #[test]
    fn jitter_edge_probabilities() {
        let never = resolve_drink_jitter(Some("0:1")).unwrap();
        let always = resolve_drink_jitter(Some("1:1")).unwrap();
        for f in 0..100 {
            assert!(!never.masks(7, f));
            assert!(always.masks(7, f));
        }
        assert!(resolve_drink_jitter(None).is_none());
    }

    #[test]
    #[should_panic(expected = "is not \"<p>:<salt>\"")]
    fn jitter_malformed_panics() {
        resolve_drink_jitter(Some("0.5"));
    }

    #[test]
    #[should_panic(expected = "is not \"<p>:<salt>\"")]
    fn jitter_out_of_range_p_panics() {
        resolve_drink_jitter(Some("1.5:2"));
    }

    #[test]
    fn swap_flag_resolution() {
        assert!(!resolve_potion_swap_flag(None, None));
        assert!(!resolve_potion_swap_flag(Some("0"), Some("/t.json")));
        assert!(resolve_potion_swap_flag(Some("1"), Some("/t.json")));
        // GH #379: an OC table is no longer required to turn the surface on.
        assert!(resolve_potion_swap_flag(Some("1"), None));
        // …it selects who decides instead.
        assert!(swap_is_rule_side(Some("/t.json")));
        assert!(!swap_is_rule_side(None));
    }

    #[test]
    #[should_panic(expected = "must be unset")]
    fn swap_flag_garbage_panics() {
        resolve_potion_swap_flag(Some("2"), Some("/t.json"));
    }

    /// The swap rule: replace the WORST held potion iff the offer prices
    /// strictly higher. GOOD prices (act 1): FirePotion 4, PowerPotion
    /// default 8 (no act-1 entry), Rock 0.
    #[test]
    fn swap_rule_replaces_worst_when_offer_is_better() {
        let t = parse_potion_oc(GOOD, "test");
        // Rock (0) is the argmin — a 4-valued Fire offer beats it.
        let belt = [PotionKind::PowerPotion, PotionKind::PotionShapedRock];
        assert_eq!(t.resolve_swap(&belt, PotionKind::FirePotion, 1), Some(1));
        // Offer worse than everything held => decline.
        let belt = [PotionKind::PowerPotion, PotionKind::BlockPotion]; // 8, 8
        assert_eq!(t.resolve_swap(&belt, PotionKind::FirePotion, 1), None);
        // Strictness: offer equal to the min => decline.
        let belt = [PotionKind::FirePotion]; // 4
        assert_eq!(t.resolve_swap(&belt, PotionKind::FirePotion, 1), None);
        // First-min on ties: two Rocks, discard slot 0.
        let belt = [PotionKind::PotionShapedRock, PotionKind::PotionShapedRock];
        assert_eq!(t.resolve_swap(&belt, PotionKind::FirePotion, 1), Some(0));
        // Empty belt cannot happen on a full-belt surface; decline safely.
        assert_eq!(t.resolve_swap(&[], PotionKind::FirePotion, 1), None);
    }

    #[test]
    #[should_panic(expected = "duplicate entry")]
    fn duplicate_entry_panics() {
        parse_potion_oc(
            r#"{"version":1,"character":"ironclad","units":"hp_equiv",
                "source":"manual","default_oc":8.0,
                "entries":[{"potion":"POTION.FIRE_POTION","act":1,"oc":2.0},
                           {"potion":"POTION.FIRE_POTION","act":1,"oc":3.0}]}"#,
            "test",
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact-by-construction literals.
    fn class_scale_resolves_and_maps_rooms() {
        let d = ClassScale::resolve(None);
        assert_eq!(d.for_room(Some(RoomType::Monster)), 1.0);
        assert_eq!(d.for_room(Some(RoomType::Elite)), 1.0);
        assert_eq!(d.for_room(Some(RoomType::Boss)), 1.0);

        let c = ClassScale::resolve(Some("3,0.5,1"));
        assert_eq!(c.for_room(Some(RoomType::Monster)), 3.0);
        assert_eq!(c.for_room(Some(RoomType::Elite)), 0.5);
        assert_eq!(c.for_room(Some(RoomType::Boss)), 1.0);
        // unknown / non-combat rooms bill as trash (the conservative side)
        assert_eq!(c.for_room(None), 3.0);
        assert_eq!(c.for_room(Some(RoomType::Unknown)), 3.0);
    }

    #[test]
    #[should_panic(expected = "exactly 3")]
    fn class_scale_rejects_wrong_arity() {
        ClassScale::resolve(Some("2,1"));
    }

    #[test]
    #[should_panic(expected = "CLASS_SCALE")]
    fn class_scale_rejects_negatives() {
        ClassScale::resolve(Some("2,-1,1"));
    }

    // ── critic mode (P1, plan 2026-08-20-ppo-potion-bootstrap) ──────────

    #[cfg(feature = "ppo")]
    #[test]
    #[allow(clippy::float_cmp)] // exact zeros by construction.
    fn critic_charges_are_nonnegative_and_monotone_in_scale() {
        // Slot 0: dV = 0.1 V-units at 0.005 V-per-HP ⇒ 20 HP-equiv,
        // ×1.5 scale ⇒ 30. Slot 1: counterfactual ABOVE the factual
        // (negative dV) must clamp to a 0 charge, never a bonus.
        let charges = critic_charges_from_values(1.0, 0.005, &[0.9, 1.05], 1.5);
        assert!(charges.iter().all(|q| *q >= 0.0 && q.is_finite()));
        assert!((charges[0] - victory_hp_equiv_to_q(30.0)).abs() < 1e-6);
        assert_eq!(charges[1], 0.0);
        assert_eq!(charges[2..], [0.0; 6]);

        let doubled = critic_charges_from_values(1.0, 0.005, &[0.9, 1.05], 3.0);
        assert!(
            (doubled[0] - 2.0 * charges[0]).abs() < 1e-6,
            "charge must be linear in STS2_RUNAI_POTION_OC_SCALE"
        );
    }

    #[cfg(feature = "ppo")]
    #[test]
    #[allow(clippy::float_cmp)] // exact zeros by construction.
    fn critic_charges_zero_on_degenerate_hp_rate() {
        // A non-positive V→HP rate cannot express HP-equivalents —
        // charge nothing rather than divide by ~0 (the resolver rejects
        // such rates; this pins the pure fn's own guard).
        assert_eq!(critic_charges_from_values(1.0, 0.0, &[0.5], 1.5), [0.0; 8]);
        assert_eq!(critic_charges_from_values(1.0, -0.2, &[0.5], 1.5), [0.0; 8]);
    }

    #[cfg(feature = "ppo")]
    #[test]
    fn critic_swap_rule_replaces_argmin_iff_offer_strictly_higher() {
        // Same rule as the table's `resolve_swap`, on critic dVs.
        assert_eq!(critic_swap_from_values(&[0.3, 0.1], 0.2), Some(1));
        assert_eq!(critic_swap_from_values(&[0.3, 0.1], 0.1), None);
        assert_eq!(critic_swap_from_values(&[0.3, 0.1], 0.05), None);
        // first-min wins ties
        assert_eq!(critic_swap_from_values(&[0.1, 0.1], 0.2), Some(0));
        assert_eq!(critic_swap_from_values(&[], 0.2), None);
    }

    #[cfg(feature = "ppo")]
    #[test]
    #[allow(clippy::float_cmp)] // parsed literals, exact by construction.
    fn resolver_critic_mode_parses_scale_with_default() {
        let Some(PotionOc::Critic(c)) =
            resolve_potion_oc(Some("critic"), Some("0"), None, Some("0.00072"))
        else {
            panic!("literal \"critic\" must resolve to critic mode");
        };
        assert_eq!(c.scale(), 1.5);
        assert_eq!(c.v_per_hp(1), 0.00072);
        assert_eq!(c.v_per_hp(2), 0.00072); // single value broadcasts
        let Some(PotionOc::Critic(c)) = resolve_potion_oc(
            Some("critic"),
            Some("0"),
            Some("2.5"),
            Some("0.00072,0.00088,0.0005"),
        ) else {
            panic!("critic mode with explicit scale");
        };
        assert_eq!(c.scale(), 2.5);
        assert_eq!(c.v_per_hp(2), 0.00088);
        assert_eq!(c.v_per_hp(9), 0.0005); // out-of-range act clamps
    }

    #[cfg(feature = "ppo")]
    #[test]
    #[should_panic(expected = "HP_RATE")]
    fn resolver_critic_mode_requires_hp_rate() {
        resolve_potion_oc(Some("critic"), Some("0"), None, None);
    }

    #[test]
    #[should_panic(expected = "HP_RATE")]
    fn resolver_rejects_hp_rate_in_table_mode() {
        resolve_potion_oc(Some("/nonexistent.json"), Some("0"), None, Some("0.00072"));
    }

    #[cfg(feature = "ppo")]
    #[test]
    #[should_panic(expected = "1 or 3")]
    fn resolver_rejects_two_value_hp_rate() {
        resolve_potion_oc(Some("critic"), Some("0"), None, Some("0.1,0.2"));
    }

    #[test]
    #[should_panic(expected = "POTION_WEIGHT=0")]
    fn resolver_critic_mode_requires_lambda_zero() {
        resolve_potion_oc(Some("critic"), None, None, Some("0.00072"));
    }

    #[cfg(feature = "ppo")]
    #[test]
    #[should_panic(expected = "POTION_OC_SCALE")]
    fn resolver_rejects_invalid_scale() {
        resolve_potion_oc(Some("critic"), Some("0"), Some("-1"), Some("0.00072"));
    }

    #[test]
    #[should_panic(expected = "POTION_OC_SCALE")]
    fn resolver_rejects_scale_in_table_mode() {
        // Checked BEFORE the table file is read, so no fixture needed:
        // the scale knob only exists in critic mode and a stray setting
        // must not silently do nothing in a sweep arm.
        resolve_potion_oc(Some("/nonexistent.json"), Some("0"), Some("2.0"), None);
    }

    /// With the ONNX runtime compiled out, critic mode must be a loud
    /// configuration error — never a silent fall-through to table mode.
    #[cfg(not(feature = "ppo"))]
    #[test]
    #[should_panic(expected = "`ppo` cargo feature")]
    fn resolver_critic_mode_requires_ppo_feature() {
        resolve_potion_oc(Some("critic"), Some("0"), None, Some("0.00072"));
    }
}
