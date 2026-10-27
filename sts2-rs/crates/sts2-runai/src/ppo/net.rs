//! tract-onnx runtime wrapper for the exported PPO `RunPolicyNet`.
//!
//! Loads the fixed-shape ONNX graph committed at
//! `sts2-pycli/data/ppo/ppo_fullrun_it350.onnx` (Task 1's exporter,
//! `export_onnx.py`) and runs raw forward passes. This is a thin shape-
//! and-dtype-only wrapper: [`PpoNet::forward_raw`] takes flat row-major
//! slices matching the exporter's `pad_to_fixed()` contract (Design §7
//! of the plan) and returns the raw `(logits, value)` outputs
//! unmodified — masked-softmax/argmax collation is Task 7's job
//! (`ppo::policy`), not this module's.
//!
//! Input order (must match the exporter's `input_names`, since tract's
//! `run()` is positional): `cat, cont, sec, pad_mask, obs_len, act_mask`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sts2_core::potions::CharacterId;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::DimLike;
use tract_onnx::tract_hir::infer::Factoid;

use super::encoder::{empty_cat, Featurized, N_CAT, N_CONT};
use super::features::Features;

/// A loaded, optimized, runnable PPO run-policy net.
pub struct PpoNet {
    model: Arc<TypedRunnableModel>,
    /// The continuous-feature width the exported graph expects (its
    /// `cont` input's last dim). Legacy champions were exported at 18;
    /// ascension-conditioned nets export at 19 ([`N_CONT`]). The encoder
    /// always emits [`N_CONT`]; [`Self::infer`] feeds only the first
    /// `cont_dim` columns, so an 18-cont champion drops the trailing
    /// ascension slot and one binary serves both.
    cont_dim: usize,
    /// The categorical-feature width the exported graph expects (its
    /// `cat` input's last dim). Champions promoted before the
    /// boss-identity input export at 9; boss-conditioned nets export at
    /// [`N_CAT`]. Read from the graph for exactly the same reason as
    /// [`Self::cont_dim`]: the encoder always emits [`N_CAT`] and
    /// [`Self::infer`] feeds only the first `cat_dim` columns, so a
    /// 9-categorical champion drops the trailing boss slot and one
    /// binary serves both. This is what lets the boss input land without
    /// re-exporting (or invalidating) the four already-gated champions.
    cat_dim: usize,
    /// Max total (observation + action) tokens this export accepts — the
    /// `T` of its `[1, T, W]` `cont` input. Read from the graph rather
    /// than duplicated as a const so a re-export at a different budget
    /// can never silently mismatch the Rust side (GH #315).
    t_max: usize,
    /// Max action slots this export accepts — the `A` of its `[1, A]`
    /// `act_mask` input. See [`Self::t_max`] for why it is derived.
    a_max: usize,
}

impl std::fmt::Debug for PpoNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PpoNet").finish_non_exhaustive()
    }
}

/// The ONNX initializer serving each categorical column, in [`Cat`] /
/// [`empty_cat`] order: `[card, upg, relic, potion, room, event, fixed,
/// optidx, surface, boss, deck_op, enchant]`.
///
/// [`Cat`]: super::encoder::Cat
const CAT_EMBEDDING_NODES: [&str; N_CAT] = [
    "net.emb_card.weight",
    "net.emb_upg.weight",
    "net.emb_relic.weight",
    "net.emb_potion.weight",
    "net.emb_room.weight",
    "net.emb_event.weight",
    "net.emb_fixed.weight",
    "net.emb_optidx.weight",
    "net.emb_surface.weight",
    "net.emb_boss.weight",
    "net.emb_deckop.weight",
    "net.emb_ench.weight",
];

/// Reject a graph whose categorical embedding tables are the wrong height
/// for the vocabularies this binary will index them with.
///
/// # Why this exists (GH #544)
///
/// Every PPO vocab is derived from the content on purpose — `ppo::vocab`
/// builds the boss registry from `run_room_gen::all_boss_encounter_ids()`,
/// the card registry from `CardId::all()`, and so on, "so the vocab cannot
/// drift from the content". The flip side is that **adding content silently
/// re-numbers every table**: the registries are sorted, so a new id inserted
/// in the middle shifts every id after it onto a different embedding row.
///
/// Two failure modes follow, and only one of them is loud:
///
/// * **Out of range** — tract's `Gather` (`ops/array/gather.rs:71`) slices
///   **unchecked**, so an index past the end panics mid-inference rather than
///   returning `Err`. Loud, but it unwinds through `run()`, so a `run_arena`
///   sweep loses that worker's labels instead of reporting a clean failure.
/// * **In range but wrong** — a shifted id that still lands inside the table
///   reads a *different* entity's embedding. The net keeps answering, and it
///   answers plausibly. Nothing downstream notices.
///
/// So a bounds check at inference time would be too late and too narrow. This
/// checks the invariant at load: [`empty_cat`] is the pad index for each
/// column (`nn.Embedding(rows, dim, padding_idx=rows - 1)`), so the table must
/// be exactly `empty_cat()[c] + 1` rows tall. Any content addition that would
/// re-number column `c` moves that number and this refuses the graph, by name,
/// before it can serve a single decision.
///
/// Only the first `cat_dim` columns are checked — a champion promoted before
/// the boss input exports `cat_dim == 9`, [`PpoNet::infer`] drops the trailing
/// boss slot, and its graph has no `emb_boss` table to check.
///
/// Sibling of the `cont_dim` / `cat_dim` / `t_max` / `a_max` reads below,
/// which exist so "a re-export at a different budget can never silently
/// mismatch the Rust side" (GH #315). The embedding tables were missed only
/// because they are initializers rather than inputs.
fn check_cat_embedding_rows(model: &InferenceModel, cat_dim: usize) -> Result<(), String> {
    let pad = empty_cat();
    for (col, node_name) in CAT_EMBEDDING_NODES.iter().enumerate().take(cat_dim) {
        // The pad id is the last valid row, so height == pad + 1.
        let want = usize::try_from(pad[col])
            .map_err(|_| format!("column {col} has a negative pad id {}", pad[col]))?
            + 1;
        let node = model
            .nodes()
            .iter()
            .find(|n| n.name == *node_name)
            .ok_or_else(|| {
                format!(
                    "graph consumes categorical column {col} but has no `{node_name}` initializer — the exporter's naming changed, and leaving the check off would let a re-numbered vocab through silently (GH #544)"
                )
            })?;
        let shape = model
            .outlet_fact(OutletId::new(node.id, 0))
            .map_err(|e| e.to_string())?
            .shape
            .concretize()
            .ok_or_else(|| format!("`{node_name}` has no concrete shape"))?;
        let got = shape
            .first()
            .and_then(|d| d.to_usize().ok())
            .ok_or_else(|| {
                format!("`{node_name}` has shape {shape:?}, expected 2-D [rows, dim]")
            })?;
        if got != want {
            return Err(format!(
                "`{node_name}` has {got} rows but this build indexes it with {want} (pad id {}). The vocab for categorical column {col} changed since this net was exported, so its embedding rows no longer mean what the net learned — re-export/re-bake the champion. See GH #544.",
                pad[col]
            ));
        }
    }
    Ok(())
}

impl PpoNet {
    /// Load and optimize the ONNX graph at `path`.
    ///
    /// # Errors
    /// Returns `Err` with tract's error text if the graph fails to
    /// parse, type/shape-infer, optimize, or compile into a runnable
    /// plan.
    pub fn load(path: &Path) -> Result<Self, String> {
        let inference = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| e.to_string())?;
        Self::from_inference(inference)
    }

    /// [`Self::load`] over the ONNX bytes themselves — the browser
    /// workbench has no filesystem: the page fetches the champion file and
    /// hands the bytes to the wasm engine. Same parse, checks and
    /// optimization as the path loader.
    ///
    /// # Errors
    /// As [`Self::load`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = std::io::Cursor::new(bytes);
        let inference = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|e| e.to_string())?;
        Self::from_inference(inference)
    }

    fn from_inference(inference: InferenceModel) -> Result<Self, String> {
        // Keep the un-optimized graph long enough to read the embedding
        // initializers by name: `into_optimized()` is free to fuse and rename
        // them, so `check_cat_embedding_rows` has to run first.
        let optimized = inference
            .clone()
            .into_optimized()
            .map_err(|e| e.to_string())?;
        // Input 1 is `cont` (positional order: cat, cont, sec, …), exported
        // at fixed shape [1, T_MAX, W]. Read W so we can serve both legacy
        // (18) and ascension-conditioned (19) champions from one encoder,
        // and T so the token budget is derived from the graph instead of
        // duplicated as a const that a re-export could silently invalidate.
        let cont_shape = optimized
            .input_fact(1)
            .map_err(|e| e.to_string())?
            .shape
            .as_concrete()
            .ok_or_else(|| "cont input lacks a concrete shape".to_string())?
            .to_vec();
        let [_, t_max, cont_dim] = cont_shape[..] else {
            return Err(format!(
                "cont input has shape {cont_shape:?}, expected 3-D [1, T_MAX, W]"
            ));
        };
        if cont_dim > N_CONT {
            return Err(format!(
                "model expects {cont_dim} cont features but encoder emits only {N_CONT}"
            ));
        }
        // Input 0 is `cat`, exported at fixed shape [1, T_MAX, N_CAT]. Read
        // its width so pre-boss-input champions (9) and boss-conditioned ones
        // (10) both load. Its T must agree with `cont`'s — they are two views
        // of the same token axis, so a mismatch is a malformed export.
        let cat_shape = optimized
            .input_fact(0)
            .map_err(|e| e.to_string())?
            .shape
            .as_concrete()
            .ok_or_else(|| "cat input lacks a concrete shape".to_string())?
            .to_vec();
        let [_, cat_t, cat_dim] = cat_shape[..] else {
            return Err(format!(
                "cat input has shape {cat_shape:?}, expected 3-D [1, T_MAX, N_CAT]"
            ));
        };
        if cat_t != t_max {
            return Err(format!(
                "cat input has T={cat_t} but cont input has T={t_max}; malformed export"
            ));
        }
        if cat_dim > N_CAT {
            return Err(format!(
                "model expects {cat_dim} categorical features but encoder emits only {N_CAT}"
            ));
        }
        // Widths agree; now check each categorical column's embedding table is
        // the right HEIGHT for the vocab that indexes it (GH #544).
        check_cat_embedding_rows(&inference, cat_dim)?;
        // Input 5 is `act_mask`, exported at fixed shape [1, A_MAX].
        let act_shape = optimized
            .input_fact(5)
            .map_err(|e| e.to_string())?
            .shape
            .as_concrete()
            .ok_or_else(|| "act_mask input lacks a concrete shape".to_string())?
            .to_vec();
        let [_, a_max] = act_shape[..] else {
            return Err(format!(
                "act_mask input has shape {act_shape:?}, expected 2-D [1, A_MAX]"
            ));
        };
        if a_max > t_max {
            return Err(format!(
                "model exports a_max={a_max} > t_max={t_max}; action tokens cannot fit"
            ));
        }
        let model = optimized.into_runnable().map_err(|e| e.to_string())?;
        Ok(Self {
            model,
            cont_dim,
            cat_dim,
            t_max,
            a_max,
        })
    }

    /// The continuous-feature width this model expects (18 legacy, 19 with
    /// the ascension input). See [`Self::cont_dim`] field docs.
    #[must_use]
    pub fn cont_dim(&self) -> usize {
        self.cont_dim
    }

    /// The categorical-feature width this model expects (9 before the
    /// boss-identity input, 10 with it). See [`Self::cat_dim`] field docs.
    #[must_use]
    pub fn cat_dim(&self) -> usize {
        self.cat_dim
    }

    /// Max total (observation + action) tokens this export accepts.
    #[must_use]
    pub fn t_max(&self) -> usize {
        self.t_max
    }

    /// Max action slots this export accepts.
    #[must_use]
    pub fn a_max(&self) -> usize {
        self.a_max
    }

    /// Run one forward pass over flat, row-major, fixed-shape inputs.
    ///
    /// Shapes: `cat` is `[t_max, cat_dim]` i64 and `cont` is
    /// `[t_max, cont_dim]` f32 — both at the model's expected width, see
    /// [`Self::cat_dim`] / [`Self::cont_dim`];
    /// `sec`/`pad_mask` are `[t_max]`, `obs_len` is a scalar, `act_mask`
    /// is `[a_max]` bool — all four budgets read off the loaded graph
    /// ([`Self::t_max`]/[`Self::a_max`]). Batch dim is always 1 and added
    /// internally.
    ///
    /// # Errors
    /// Returns `Err` with tract's error text if tensor construction or
    /// the forward pass fails (e.g. a slice length mismatch).
    pub fn forward_raw(
        &self,
        cat: &[i64],
        cont: &[f32],
        sec: &[i64],
        pad_mask: &[bool],
        obs_len: i64,
        act_mask: &[bool],
    ) -> Result<(Vec<f32>, f32), String> {
        let t = self.t_max;
        let inputs = tvec![
            TValue::from(
                Tensor::from_shape(&[1, t, self.cat_dim], cat).map_err(|e| e.to_string())?,
            ),
            TValue::from(
                Tensor::from_shape(&[1, t, self.cont_dim], cont).map_err(|e| e.to_string())?,
            ),
            TValue::from(Tensor::from_shape(&[1, t], sec).map_err(|e| e.to_string())?),
            TValue::from(Tensor::from_shape(&[1, t], pad_mask).map_err(|e| e.to_string())?),
            TValue::from(Tensor::from_shape(&[1], &[obs_len]).map_err(|e| e.to_string())?),
            TValue::from(
                Tensor::from_shape(&[1, self.a_max], act_mask).map_err(|e| e.to_string())?,
            ),
        ];
        let out = self.model.run(inputs).map_err(|e| e.to_string())?;
        let logits = out[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| e.to_string())?
            .iter()
            .copied()
            .collect::<Vec<f32>>();
        let value = *out[1]
            .to_plain_array_view::<f32>()
            .map_err(|e| e.to_string())?
            .iter()
            .next()
            .ok_or("empty value output")?;
        Ok((logits, value))
    }

    /// End-to-end inference: pad `feat` to this export's fixed
    /// `(t_max, a_max)` shape (mirrors `export_onnx.pad_to_fixed` bit-for-bit —
    /// pad rows carry [`empty_cat`] ids / zero cont / `sec=0` /
    /// `pad_mask=true`, `pad_mask` cleared over `[0, o+a)`, `act_mask`
    /// true over `[0, n_actions)`), run [`Self::forward_raw`], and
    /// collate the masked logits into a softmax distribution over the
    /// real actions plus the scalar value.
    ///
    /// Softmax is computed in `f64` (max-subtract, exp, normalize) to
    /// mirror torch `F.softmax` numerics, then cast back to `f32`.
    ///
    /// # Errors
    /// Returns `Err` if `obs_len + n_actions` exceeds [`Self::t_max`] or
    /// `n_actions` exceeds [`Self::a_max`] (export budget overflow —
    /// never truncated), or if the underlying [`Self::forward_raw`] call
    /// fails. The overflow text names the budget it violated as well as
    /// the demand, so a caller can tell which of the two axes bound
    /// without knowing which champion is loaded (GH #315).
    pub fn infer(&self, feat: &Featurized) -> Result<(Vec<f32>, f32), String> {
        let o = feat.obs_cat.len();
        let a = feat.n_actions;
        let (t_max, a_max) = (self.t_max, self.a_max);
        if o + a > t_max || a > a_max {
            return Err(format!(
                "decision exceeds export budget: obs={o} act={a} (t_max={t_max} a_max={a_max})"
            ));
        }

        let pad_row = empty_cat();
        let w = self.cont_dim;
        // Feed only the first `cat_w` categorical columns for the same reason
        // as `w` below: the encoder emits N_CAT (boss identity in the trailing
        // slot), so a 9-categorical champion has cat_w = 9, that slot is
        // dropped, and the legacy model is unchanged.
        let cat_w = self.cat_dim;
        let mut cat = vec![0i64; t_max * cat_w];
        let mut cont = vec![0f32; t_max * w];
        let mut sec = vec![0i64; t_max];
        for t in 0..t_max {
            let row: &[i64; N_CAT] = if t < o {
                &feat.obs_cat[t]
            } else if t < o + a {
                &feat.act_cat[t - o]
            } else {
                &pad_row
            };
            cat[t * cat_w..(t + 1) * cat_w].copy_from_slice(&row[..cat_w]);
        }
        // Feed only the first `w` continuous columns. The encoder emits
        // N_CONT (ascension in the trailing slot); an 18-cont champion has
        // w = 18, so that slot is dropped and the legacy model is unchanged.
        for (t, row) in feat.obs_cont.iter().enumerate() {
            cont[t * w..(t + 1) * w].copy_from_slice(&row[..w]);
        }
        for (i, row) in feat.act_cont.iter().enumerate() {
            let t = o + i;
            cont[t * w..(t + 1) * w].copy_from_slice(&row[..w]);
        }
        sec[..o].copy_from_slice(&feat.obs_sec);
        sec[o..o + a].copy_from_slice(&feat.act_sec);

        let mut pad_mask = vec![true; t_max];
        pad_mask[..o + a].fill(false);
        let mut act_mask = vec![false; a_max];
        act_mask[..a].fill(true);

        let (logits, value) =
            self.forward_raw(&cat, &cont, &sec, &pad_mask, o as i64, &act_mask)?;

        let real = &logits[..a];
        let max = f64::from(real.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let exps: Vec<f64> = real.iter().map(|&l| (f64::from(l) - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let probs = exps.into_iter().map(|e| (e / sum) as f32).collect();

        Ok((probs, value))
    }
}

/// The committed sibling-checkout ONNX filename for each character's
/// deployed champion.
///
/// **The match arms below are generated.** `sts2-pycli/data/ppo/champions.toml`
/// is the single source of truth for which net each character serves; the
/// Python resolver reads it directly, and `just sync-champions` rewrites this
/// region and the Dockerfile `COPY` block from it (`just check-champions` is
/// the gate). Edit the manifest, never the arms.
///
/// This crate does *not* read the manifest at build time on purpose: it is
/// built from standalone worktrees where no sibling `sts2-pycli/` exists — the
/// same reason [`onnx_model_path_for`] guards with `sibling.exists()` — and an
/// `include_str!` would make those worktrees unbuildable.
///
/// Per-net provenance (what gate promoted it, against which baseline, at what
/// significance) lives in `sts2-pycli/data/ppo/<net>.provenance.md`, with a
/// one-line summary on each manifest entry.
///
/// Total: every character has a trained champion. This returned
/// `Option<&str>` while a sixth `CharacterId` variant existed to return
/// `None` for; that variant is gone, and the `None` case with it.
#[must_use]
pub fn champion_onnx_filename(character: CharacterId) -> &'static str {
    champion_entry(character).0
}

/// `(onnx filename, STS2_PPO_FEATURES spec)` for each character's champion.
///
/// The feature spec rides alongside the filename because it is part of the
/// same fact: *which net this character serves, and at which encoder schema
/// it was trained*. Splitting them across two generated regions is how they
/// would drift.
///
/// See [`champion_onnx_filename`] for why the arms are generated rather than
/// read from the manifest at runtime.
#[must_use]
fn champion_entry(character: CharacterId) -> (&'static str, &'static str) {
    match character {
        // GENERATED FROM sts2-pycli/data/ppo/champions.toml -- DO NOT EDIT
        CharacterId::Defect => ("ppo_fullrun_defect_deckmut_it150_v0111vocab.onnx", ""),
        CharacterId::Ironclad => (
            "ppo_fullrun_ironclad_cardmods_it850.onnx",
            "card_mods,reward_upgrade",
        ),
        CharacterId::Necrobinder => ("ppo_fullrun_necrobinder_deckmut_it150_v0111vocab.onnx", ""),
        CharacterId::Regent => ("ppo_fullrun_regent_deckmut_it150_v0111vocab.onnx", ""),
        CharacterId::Silent => ("ppo_fullrun_silent_cont_it300_v0111vocab.onnx", ""),
        // END GENERATED CHAMPIONS
    }
}

/// The encoder feature set `character`'s served net was TRAINED with.
///
/// **This is what makes a mixed fleet serveable.** `STS2_PPO_FEATURES` is
/// parsed once per process ([`Features::active`]), so one process serving all
/// five champions cannot express "on for Ironclad, off for the other four" —
/// and handing a champion a feature it never trained on is undetectable from
/// the tensor, because a gated feature changes what goes into an EXISTING slot
/// rather than adding a column (see [`super::features`]). Every champion to
/// date holds an unconditional anti-upgraded-card prior that would run
/// BACKWARDS if `reward_upgrade` were switched on underneath it.
///
/// So the manifest wins for a committed champion, and the env var is scoped to
/// the one case the manifest cannot describe:
///
/// * `STS2_RUNAI_PPO` set — an explicit net at an arbitrary path, which no
///   manifest entry describes. This is the gate-sweep / pre-promotion lane,
///   and `STS2_PPO_FEATURES` is how you declare its schema.
/// * otherwise — the committed champion, at the spec `champions.toml` records
///   for it. `STS2_PPO_FEATURES` is deliberately IGNORED here: it is
///   process-wide and the fleet is not.
#[must_use]
pub fn features_for(character: CharacterId) -> Features {
    if std::env::var_os("STS2_RUNAI_PPO").is_some() {
        return Features::active();
    }
    Features::parse(champion_entry(character).1)
}

/// Resolve the committed ONNX model path for `character`: the
/// `STS2_RUNAI_PPO` env var if set (an explicit override that wins for
/// every character), else the sibling-checkout committed champion
/// (`<manifest>/../../../sts2-pycli/data/ppo/<champion>.onnx`) if it
/// exists on disk. Returns `None` (never an error) so callers — like the
/// smoke test — can skip gracefully when the champion file is absent from
/// this checkout.
#[must_use]
pub fn onnx_model_path_for(character: CharacterId) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("STS2_RUNAI_PPO") {
        return Some(PathBuf::from(p));
    }
    let filename = champion_onnx_filename(character);
    let sibling = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../sts2-pycli/data/ppo")
        .join(filename);
    sibling.exists().then_some(sibling)
}

/// Resolve the committed ONNX model path for the default character
/// (Ironclad). Back-compat shim for callers that predate per-character
/// resolution (the parity/smoke tests and the PyO3 agent); new code
/// should thread the character through [`onnx_model_path_for`].
#[must_use]
pub fn onnx_model_path() -> Option<PathBuf> {
    onnx_model_path_for(CharacterId::Ironclad)
}
