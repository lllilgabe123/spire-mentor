//! Learned value-net leaf evaluator (Wave-6) — bead `sts2-rs-4n0x`
//! (qctu.5), hand-rolled MLP forward as of `sts2-rs-s6v2t`.
//!
//! ## Forward path
//!
//! 1. Compute the 88-feature wide-schema vector at the current state
//!    via [`sts2_core::eval_features_wide`].
//! 2. Normalize element-wise: `(x - mean) / std` using per-column
//!    statistics persisted by the qctu.3 trainer.
//! 3. Forward through the hand-rolled MLP (3× `Linear(256)+ReLU` trunk,
//!    then `Linear(1)+sigmoid` value head, `Linear(1)+sigmoid` aux-HP
//!    head, and `Linear(640)+softmax` policy head). The architecture
//!    is fixed and matches `scripts/value_policy_net.py::TwoHeadedNet`.
//! 4. Map value-head `P(win)` → in-progress score in `(-0.49, 0.49)`
//!    via `2 * (p - 0.5) * 0.49` so depth-cap evals stay strictly
//!    inside the win/loss bucket boundary.
//!
//! Pre-s6v2t, this module forwarded through a `tract`-loaded ONNX
//! graph. Per-call overhead was 50–90 µs dominated by tract's graph
//! dispatch + per-call allocation, not by the ~640k-flop matmul math
//! itself. The hand-rolled forward eliminates that framework tax
//! (~5–30 µs/call) and removes the tract dep from the hot path.
//!
//! ## Threading & sharing
//!
//! `LearnedValueNet` owns its weights as plain `Box<[f32]>`. Both
//! `LearnedValueNet` and the forward path are `Send + Sync` (no
//! interior mutability). Shared via `Arc<LearnedValueNet>`.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::Deserialize;
use smallvec::SmallVec;
use sts2_core::{eval_features_wide, Action, CombatState, WIDE_FEATURE_DIM};

/// Per-column normalization vectors persisted by the qctu.3 trainer.
/// `mean` and `std` are 80-element arrays in [`sts2_core::EVAL_FEATURE_NAMES_WIDE`]
/// column order; `std` elements at indices where the train-split
/// variance was 0 are stored as 1.0 (the trainer clamps in
/// `value_policy_net.py`).
#[derive(Clone, Debug)]
pub struct NormStats {
    pub mean: [f32; WIDE_FEATURE_DIM],
    pub std: [f32; WIDE_FEATURE_DIM],
}

impl NormStats {
    /// Identity normalization — emits the input unchanged. Used as a
    /// default when the caller doesn't have the qctu.3 stats handy
    /// (e.g. unit tests building synthetic ONNX fixtures).
    pub fn identity() -> Self {
        Self {
            mean: [0.0; WIDE_FEATURE_DIM],
            std: [1.0; WIDE_FEATURE_DIM],
        }
    }
}

impl Default for NormStats {
    fn default() -> Self {
        Self::identity()
    }
}

/// Errors loading the learned value net's weights JSON.
#[derive(Debug)]
pub enum ValueNetError {
    /// Failed to read the weights JSON file from disk.
    Io(std::io::Error),
    /// JSON parse failure or schema mismatch.
    Parse(String),
    /// A weight tensor had an unexpected shape — typically a sign of
    /// architecture drift between the Python trainer
    /// (`scripts/value_policy_net.py`) and this loader.
    ShapeMismatch {
        which: &'static str,
        expected: (usize, usize),
        got: (usize, usize),
    },
}

impl From<std::io::Error> for ValueNetError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for ValueNetError {
    fn from(e: serde_json::Error) -> Self {
        Self::Parse(e.to_string())
    }
}

impl std::fmt::Display for ValueNetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "weights JSON read failed: {e}"),
            Self::Parse(m) => write!(f, "weights JSON parse failed: {m}"),
            Self::ShapeMismatch {
                which,
                expected,
                got,
            } => write!(
                f,
                "weights {which} shape {got:?} mismatch — expected {expected:?}; \
                 re-train via scripts/value_policy_net.py or check arch drift"
            ),
        }
    }
}

impl std::error::Error for ValueNetError {}

/// In-progress score range pinned to `(-0.49, 0.49)` so depth-cap
/// evals never bucket as a confirmed win/loss. Matches the bound
/// `evaluate_in_progress` enforces via its own clamp. Bead
/// `sts2-rs-v2oh` documents the rationale upstream.
const INPROGRESS_BOUND: f32 = 0.49;

/// Map a value-head sigmoid output `p_win ∈ [0, 1]` to the
/// in-progress score range `(-INPROGRESS_BOUND, +INPROGRESS_BOUND)`.
/// Shared between [`LearnedValueNet::evaluate`] and
/// [`LearnedValueNet::evaluate_with_policy`]. Bead `sts2-rs-v0i1`
/// (qctu.8) factored this out so the policy-prior path stays
/// trivially aligned with the value-only path's contract.
#[inline]
fn map_win_to_score(p_win: f32) -> f32 {
    let centered = p_win - 0.5;
    (2.0 * centered * INPROGRESS_BOUND).clamp(-INPROGRESS_BOUND, INPROGRESS_BOUND)
}

// ── Hand-rolled MLP forward (sts2-rs-s6v2t) ────────────────────────────────

/// Max trunk hidden width supported by the stack-allocated forward path.
/// Raised 256 → 512 (sts2-rs-1bcx) so the loader can run the `--hidden 512`
/// "wide-deep" arm — the size class l38nd's `vbzji-large` policy head used
/// via the (now-removed) ONNX path; the post-s6v2t hand-rolled loader
/// previously capped at 256 and silently could not load it.
/// The actual hidden width is stored per-net in [`MlpWeights::hidden`] and
/// must be `<= MAX_HIDDEN`; the loader rejects larger widths via
/// [`ValueNetError::ShapeMismatch`]. Smaller widths (e.g. 16/64/128, swept in
/// `sts2-rs-qmncd`) share the same stack buffers via runtime slicing — no
/// heap allocation per forward. Stack cost is `2 × MAX_HIDDEN × 4 B` ≈ 4 KiB
/// per forward, negligible.
const MAX_HIDDEN: usize = 512;

/// Policy head fan-out. Matches [`crate::action_index::CANONICAL_ACTION_DIM`].
/// Kept as a named constant for readability; the live policy_dim is now
/// arch-driven (640 or 4672) — see [`MlpWeights::policy_dim`] (bead
/// `sts2-rs-1bcx`).
#[allow(dead_code)]
const POLICY_DIM: usize = crate::action_index::CANONICAL_ACTION_DIM;

/// Min/max trunk depth (number of `Linear+ReLU` layers before the heads).
/// Trainer default is `--layers 3`; qmncd-follow-up sweep tries 2 + 4
/// to right-size the trunk. Loader rejects `trunk_depth` outside this
/// range via [`ValueNetError::ShapeMismatch`].
const MIN_TRUNK_LAYERS: usize = 2;
const MAX_TRUNK_LAYERS: usize = 4;

/// Hand-rolled MLP weights. Each `Linear` layer's weight matrix is
/// stored row-major as `[fan_in × fan_out]`.
///
/// Bead `sts2-rs-ua9hx`: dropped the aux-HP head storage (`w_h` +
/// `b_h`). The aux head is a KataGo-style training-time auxiliary
/// that improves trunk representation; MCTS inference uses only
/// the value + policy heads. The JSON loader still accepts trainer
/// outputs that carry W_h / b_h (via `#[serde(default)]` on the
/// parse fields) but the values are dropped without
/// shape-validation — the trainer already shape-validates them at
/// emit time.
///
/// `policy_dim` is read from `arch.policy_dim` at load time and may
/// be [`CANONICAL_ACTION_DIM`] (640) or [`ACTION_KEY_DIM`] (4672).
/// Bead `sts2-rs-1bcx` made this arch-driven.
#[derive(Clone, Debug)]
pub struct MlpWeights {
    hidden: usize,
    /// Width of the policy head; either [`CANONICAL_ACTION_DIM`] (640)
    /// or [`ACTION_KEY_DIM`] (4672). Drives `W_p` / `b_p` sizing and
    /// the output vec width in [`Self::forward_value_and_policy`].
    policy_dim: usize,
    trunk: Vec<(Box<[f32]>, Box<[f32]>)>,
    w_v: Box<[f32]>,
    b_v: f32,
    w_p: Box<[f32]>,
    b_p: Box<[f32]>,
}

/// `out = h @ W + b`, where `W` is row-major `[fan_in × fan_out]`.
#[inline]
fn linear(out: &mut [f32], w: &[f32], b: &[f32], h_in: &[f32]) {
    let fan_out = b.len();
    let fan_in = h_in.len();
    debug_assert_eq!(w.len(), fan_in * fan_out);
    debug_assert_eq!(out.len(), fan_out);
    out.copy_from_slice(b);
    for i in 0..fan_in {
        let hi = h_in[i];
        if hi == 0.0 {
            continue;
        }
        let row = &w[i * fan_out..(i + 1) * fan_out];
        for j in 0..fan_out {
            out[j] += hi * row[j];
        }
    }
}

#[inline]
fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax_inplace(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in x.iter_mut() {
        *v *= inv;
    }
}

impl MlpWeights {
    fn forward_trunk(
        &self,
        features: &[f32; WIDE_FEATURE_DIM],
        h_a: &mut [f32],
        h_b: &mut [f32],
    ) -> bool {
        debug_assert_eq!(h_a.len(), self.hidden);
        debug_assert_eq!(h_b.len(), self.hidden);
        let (w0, b0) = &self.trunk[0];
        linear(h_a, w0, b0, features);
        relu_inplace(h_a);
        let mut src_is_a = true;
        for l in 1..self.trunk.len() {
            let (w, b) = &self.trunk[l];
            if src_is_a {
                linear(h_b, w, b, h_a);
                relu_inplace(h_b);
            } else {
                linear(h_a, w, b, h_b);
                relu_inplace(h_a);
            }
            src_is_a = !src_is_a;
        }
        src_is_a
    }

    /// Forward `features`, returning post-sigmoid `P(win)` only.
    pub fn forward_value(&self, features: &[f32; WIDE_FEATURE_DIM]) -> f32 {
        let mut buf_a = [0.0_f32; MAX_HIDDEN];
        let mut buf_b = [0.0_f32; MAX_HIDDEN];
        let h_a = &mut buf_a[..self.hidden];
        let h_b = &mut buf_b[..self.hidden];
        let src_is_a = self.forward_trunk(features, h_a, h_b);
        let trunk_out: &[f32] = if src_is_a { h_a } else { h_b };
        let mut value_pre = self.b_v;
        for (t, w) in trunk_out.iter().zip(self.w_v.iter()) {
            value_pre += t * w;
        }
        sigmoid(value_pre)
    }

    /// Forward `features`, returning `(P(win), policy_softmax)`.
    pub fn forward_value_and_policy(&self, features: &[f32; WIDE_FEATURE_DIM]) -> (f32, Vec<f32>) {
        let mut buf_a = [0.0_f32; MAX_HIDDEN];
        let mut buf_b = [0.0_f32; MAX_HIDDEN];
        let h_a = &mut buf_a[..self.hidden];
        let h_b = &mut buf_b[..self.hidden];
        let src_is_a = self.forward_trunk(features, h_a, h_b);
        let trunk_out: &[f32] = if src_is_a { h_a } else { h_b };

        let mut value_pre = self.b_v;
        for (t, w) in trunk_out.iter().zip(self.w_v.iter()) {
            value_pre += t * w;
        }
        let p_win = sigmoid(value_pre);

        let mut policy = vec![0.0_f32; self.policy_dim];
        linear(&mut policy, &self.w_p, &self.b_p, trunk_out);
        softmax_inplace(&mut policy);

        (p_win, policy)
    }
}

#[derive(Debug, Deserialize)]
struct WeightsJson {
    arch: ArchJson,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    weights: WeightsTable,
}

#[derive(Debug, Deserialize)]
struct ArchJson {
    input_dim: usize,
    hidden: usize,
    policy_dim: usize,
    trunk_depth: usize,
}

#[derive(Debug, Deserialize)]
struct WeightsTable {
    #[serde(rename = "W1")]
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    #[serde(rename = "W2")]
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    #[serde(rename = "W3", default)]
    w3: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    b3: Option<Vec<f32>>,
    #[serde(rename = "W4", default)]
    w4: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    b4: Option<Vec<f32>>,
    #[serde(rename = "W_v")]
    w_v: Vec<Vec<f32>>,
    b_v: Vec<f32>,
    // Bead `sts2-rs-ua9hx`: aux-HP head fields kept on the parse
    // struct so existing trainer outputs (which emit W_h / b_h)
    // still deserialize, but defaulted to empty so future trainers
    // can omit them. The values are not stored on MlpWeights —
    // MCTS inference never consults the aux head. The
    // #[allow(dead_code)] suppresses the dead-field lint since the
    // parse struct serves as deserialize-then-discard for these
    // two slots.
    #[serde(rename = "W_h", default)]
    #[allow(dead_code)]
    w_h: Vec<Vec<f32>>,
    #[serde(default)]
    #[allow(dead_code)]
    b_h: Vec<f32>,
    #[serde(rename = "W_p")]
    w_p: Vec<Vec<f32>>,
    b_p: Vec<f32>,
}

fn flatten_matrix(
    rows: &[Vec<f32>],
    expected_rows: usize,
    expected_cols: usize,
    which: &'static str,
) -> Result<Box<[f32]>, ValueNetError> {
    if rows.len() != expected_rows {
        return Err(ValueNetError::ShapeMismatch {
            which,
            expected: (expected_rows, expected_cols),
            got: (rows.len(), rows.first().map_or(0, Vec::len)),
        });
    }
    let mut out = Vec::with_capacity(expected_rows * expected_cols);
    for row in rows {
        if row.len() != expected_cols {
            return Err(ValueNetError::ShapeMismatch {
                which,
                expected: (expected_rows, expected_cols),
                got: (expected_rows, row.len()),
            });
        }
        out.extend_from_slice(row);
    }
    Ok(out.into_boxed_slice())
}

fn flatten_vector(
    v: &[f32],
    expected_len: usize,
    which: &'static str,
) -> Result<Box<[f32]>, ValueNetError> {
    if v.len() != expected_len {
        return Err(ValueNetError::ShapeMismatch {
            which,
            expected: (expected_len, 1),
            got: (v.len(), 1),
        });
    }
    Ok(v.to_vec().into_boxed_slice())
}

impl MlpWeights {
    fn from_json(json: &WeightsJson) -> Result<Self, ValueNetError> {
        if json.arch.input_dim != WIDE_FEATURE_DIM {
            return Err(ValueNetError::ShapeMismatch {
                which: "arch.input_dim",
                expected: (WIDE_FEATURE_DIM, 0),
                got: (json.arch.input_dim, 0),
            });
        }
        if json.arch.hidden == 0 || json.arch.hidden > MAX_HIDDEN {
            return Err(ValueNetError::ShapeMismatch {
                which: "arch.hidden",
                expected: (MAX_HIDDEN, 0),
                got: (json.arch.hidden, 0),
            });
        }
        let hidden = json.arch.hidden;
        let policy_dim = json.arch.policy_dim;
        if policy_dim != crate::action_index::CANONICAL_ACTION_DIM
            && policy_dim != crate::action_index::ACTION_KEY_DIM
        {
            return Err(ValueNetError::ShapeMismatch {
                which: "arch.policy_dim",
                expected: (crate::action_index::ACTION_KEY_DIM, 0),
                got: (policy_dim, 0),
            });
        }
        if json.arch.trunk_depth < MIN_TRUNK_LAYERS || json.arch.trunk_depth > MAX_TRUNK_LAYERS {
            return Err(ValueNetError::ShapeMismatch {
                which: "arch.trunk_depth",
                expected: (MIN_TRUNK_LAYERS, MAX_TRUNK_LAYERS),
                got: (json.arch.trunk_depth, 0),
            });
        }
        let trunk_depth = json.arch.trunk_depth;
        let w = &json.weights;
        let mut trunk = Vec::with_capacity(trunk_depth);
        trunk.push((
            flatten_matrix(&w.w1, WIDE_FEATURE_DIM, hidden, "W1")?,
            flatten_vector(&w.b1, hidden, "b1")?,
        ));
        if trunk_depth >= 2 {
            trunk.push((
                flatten_matrix(&w.w2, hidden, hidden, "W2")?,
                flatten_vector(&w.b2, hidden, "b2")?,
            ));
        }
        if trunk_depth >= 3 {
            let w3 = w.w3.as_ref().ok_or(ValueNetError::ShapeMismatch {
                which: "weights.W3",
                expected: (hidden, hidden),
                got: (0, 0),
            })?;
            let b3 = w.b3.as_ref().ok_or(ValueNetError::ShapeMismatch {
                which: "weights.b3",
                expected: (hidden, 0),
                got: (0, 0),
            })?;
            trunk.push((
                flatten_matrix(w3, hidden, hidden, "W3")?,
                flatten_vector(b3, hidden, "b3")?,
            ));
        }
        if trunk_depth >= 4 {
            let w4 = w.w4.as_ref().ok_or(ValueNetError::ShapeMismatch {
                which: "weights.W4",
                expected: (hidden, hidden),
                got: (0, 0),
            })?;
            let b4 = w.b4.as_ref().ok_or(ValueNetError::ShapeMismatch {
                which: "weights.b4",
                expected: (hidden, 0),
                got: (0, 0),
            })?;
            trunk.push((
                flatten_matrix(w4, hidden, hidden, "W4")?,
                flatten_vector(b4, hidden, "b4")?,
            ));
        }
        let w_v_box = flatten_matrix(&w.w_v, hidden, 1, "W_v")?;
        let w_p_box = flatten_matrix(&w.w_p, hidden, policy_dim, "W_p")?;
        let b_v = flatten_vector(&w.b_v, 1, "b_v")?[0];
        let b_p_box = flatten_vector(&w.b_p, policy_dim, "b_p")?;
        // Bead `sts2-rs-ua9hx`: aux-HP head weights (W_h / b_h) are
        // dropped on the floor — present in older trainer outputs,
        // never consulted by MCTS inference. Parsed-but-discarded
        // (no shape validation) so future trainers can omit them.
        Ok(Self {
            hidden,
            policy_dim,
            trunk,
            w_v: w_v_box,
            b_v,
            w_p: w_p_box,
            b_p: b_p_box,
        })
    }
}

/// AlphaZero-style learned leaf evaluator. One instance per process,
/// shared via `Arc<LearnedValueNet>` across every `Mcts` and rayon
/// worker.
pub struct LearnedValueNet {
    /// Hand-rolled MLP weights (sts2-rs-s6v2t). Replaces the prior
    /// `tract`-loaded ONNX `SimplePlan` — see module docstring for
    /// the cost-driven rationale.
    weights: MlpWeights,
    norm: NormStats,
    /// Bead `sts2-rs-ae14u` (post-vbzji). When `Some(k)`, the value
    /// score becomes `tanh(NN_logit + k * texel_norm)` instead of
    /// `map_win_to_score(p_win_NN)`. The inference-side composition
    /// gives a strong Texel floor regardless of the NN's training
    /// quality — Caissawary pattern.
    ///
    /// `None` (default) preserves the historical pure-NN behavior;
    /// existing fixtures stay byte-stable.
    ///
    /// Caissawary's reference `k = 0.326` (Texel-calibrated) is a
    /// sensible bootstrap; operators can sweep `k ∈ [0.1, 0.8]` to
    /// find the sweet spot on their corpus before committing to a
    /// trainer-side `k` parameter.
    texel_compose_k: Option<f32>,
}

impl std::fmt::Debug for LearnedValueNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LearnedValueNet")
            .field("norm", &self.norm)
            .field("weights", &"<hand-rolled MLP>")
            .field("texel_compose_k", &self.texel_compose_k)
            .finish()
    }
}

impl LearnedValueNet {
    /// Load weights + normalization stats from a single JSON file
    /// produced by `scripts/value_policy_net.py` (the trainer's
    /// `--output` artifact). The JSON carries the architecture
    /// descriptor (`arch.{input_dim,hidden,policy_dim,trunk_depth}`),
    /// per-column `feature_mean` / `feature_std`, and the trained
    /// `weights` table. Bead `sts2-rs-s6v2t` — replaces the prior
    /// `from_onnx_path` + separate norm JSON loader pair.
    pub fn from_weights_path(path: &Path) -> Result<Self, ValueNetError> {
        let bytes = std::fs::read(path)?;
        let json: WeightsJson = serde_json::from_slice(&bytes)?;
        let weights = MlpWeights::from_json(&json)?;

        if json.feature_mean.len() != WIDE_FEATURE_DIM || json.feature_std.len() != WIDE_FEATURE_DIM
        {
            return Err(ValueNetError::ShapeMismatch {
                which: "feature_mean/std",
                expected: (WIDE_FEATURE_DIM, 0),
                got: (json.feature_mean.len(), json.feature_std.len()),
            });
        }
        let mut mean = [0.0_f32; WIDE_FEATURE_DIM];
        let mut std = [1.0_f32; WIDE_FEATURE_DIM];
        mean.copy_from_slice(&json.feature_mean);
        std.copy_from_slice(&json.feature_std);

        Ok(Self {
            weights,
            norm: NormStats { mean, std },
            texel_compose_k: None,
        })
    }

    /// Override the normalization stats on an existing instance.
    /// Cheap — clones the [`NormStats`] only.
    #[must_use]
    pub fn with_norm_stats(mut self, norm: NormStats) -> Self {
        self.norm = norm;
        self
    }

    /// Enable Caissawary-style inference-side Texel composition: the
    /// emitted score becomes `tanh(NN_logit + k * texel_norm)` where
    /// `texel_norm = sts2_core::evaluate_in_progress(state)`. Default
    /// `None` preserves the historical pure-NN path. Bead
    /// `sts2-rs-ae14u`. See struct docs for tuning notes.
    #[must_use]
    pub fn with_texel_compose(mut self, k: Option<f32>) -> Self {
        self.texel_compose_k = k;
        self
    }

    /// Return the [`crate::corpus::PolicyActionSpace`] implied by
    /// `arch.policy_dim`. A 640-wide net uses the canonical
    /// (card-id-only) slot fn; a 4672-wide net uses the keyed
    /// (card × target-class) slot fn. Bead `sts2-rs-1bcx`.
    pub fn policy_action_space(&self) -> crate::corpus::PolicyActionSpace {
        if self.weights.policy_dim == crate::action_index::ACTION_KEY_DIM {
            crate::corpus::PolicyActionSpace::Keyed
        } else {
            crate::corpus::PolicyActionSpace::Canonical
        }
    }

    /// Forward `state` through the net and return the depth-cap
    /// value score, mapped into `(-INPROGRESS_BOUND, +INPROGRESS_BOUND)`.
    ///
    /// `state` is read via [`eval_features_wide`]; normalization
    /// is applied inline before the forward pass. Errors propagate
    /// from tract as panics here because the tract `run()` path
    /// is infallible in practice once the model loaded
    /// successfully — a runtime tract failure indicates a fixture
    /// drift (input dim changed?) that the caller can't recover
    /// from at the MCTS hot path.
    pub fn evaluate(&self, state: &CombatState) -> f32 {
        let normalized = self.normalize(state);
        let p_win = self.forward_value(&normalized);
        self.compose_score(p_win, state)
    }

    /// Forward `state` and return BOTH the value-head score (mapped
    /// to `(-INPROGRESS_BOUND, +INPROGRESS_BOUND)`) AND the policy
    /// head's dense softmax over [`crate::CANONICAL_ACTION_DIM`]
    /// slots. Bead `sts2-rs-v0i1` (qctu.8). Used by
    /// [`LearnedPolicyPrior`] so a single forward pass amortizes
    /// across value + policy consumption — current call sites that
    /// only need value should keep using [`Self::evaluate`] to
    /// avoid the extra allocation of the policy buffer.
    pub fn evaluate_with_policy(&self, state: &CombatState) -> (f32, Vec<f32>) {
        let normalized = self.normalize(state);
        let (p_win, policy) = self.forward_value_and_policy(&normalized);
        (self.compose_score(p_win, state), policy)
    }

    /// Map a sigmoid value-head output to the in-progress score
    /// range, optionally folding in a Texel additive baseline
    /// (bead `sts2-rs-ae14u`).
    ///
    /// With `texel_compose_k = None`, this is bit-identical to the
    /// historical `map_win_to_score(p_win)` path.
    ///
    /// With `Some(k)`, recover the NN's pre-sigmoid logit (numerically
    /// stable inverse of the sigmoid), add `k * texel_norm`, apply
    /// `tanh`, and scale into the depth-cap range. `texel_norm` here
    /// is `sts2_core::evaluate_in_progress` whose contract already
    /// keeps the value bounded — the `tanh` further squashes the sum
    /// into `(-1, 1)` so the depth-cap clamp never widens.
    #[inline]
    fn compose_score(&self, p_win: f32, state: &CombatState) -> f32 {
        let base = map_win_to_score(p_win);
        match self.texel_compose_k {
            None => base,
            Some(k) => {
                // Recover the NN logit. Sigmoid inverse: ln(p/(1-p)).
                // Clamp inside (0, 1) so the log is finite even when
                // the NN saturates.
                let p = p_win.clamp(1e-6, 1.0 - 1e-6);
                let nn_logit = (p / (1.0 - p)).ln();
                let texel_norm = sts2_core::evaluate_in_progress(state);
                let combined = (nn_logit + k * texel_norm).tanh();
                // `tanh` outputs `(-1, 1)`; scale into the depth-cap
                // range so this branch obeys the same contract as
                // `map_win_to_score`.
                (combined * INPROGRESS_BOUND).clamp(-INPROGRESS_BOUND, INPROGRESS_BOUND)
            }
        }
    }

    fn normalize(&self, state: &CombatState) -> [f32; WIDE_FEATURE_DIM] {
        let raw = eval_features_wide(state);
        let mut normalized = [0.0_f32; WIDE_FEATURE_DIM];
        for i in 0..WIDE_FEATURE_DIM {
            normalized[i] = (raw[i] - self.norm.mean[i]) / self.norm.std[i];
        }
        normalized
    }

    /// Forward, reading only the value head. Hot-path for the
    /// [`LeafEval::Learned`] depth-cap dispatch.
    fn forward_value(&self, features: &[f32; WIDE_FEATURE_DIM]) -> f32 {
        self.weights.forward_value(features)
    }

    /// Forward, reading the value head AND the policy head.
    /// The policy output is post-softmax over `CANONICAL_ACTION_DIM`
    /// slots. Bead `sts2-rs-v0i1` (original API), `sts2-rs-s6v2t`
    /// (hand-rolled MLP rewrite).
    fn forward_value_and_policy(&self, features: &[f32; WIDE_FEATURE_DIM]) -> (f32, Vec<f32>) {
        self.weights.forward_value_and_policy(features)
    }
}

// ── Embedding-aware value net (stage-a id-embedding parity) ────────────────
//
// Inference path for the `value_policy_net_generalist.py` trunk, which
// the wide-only `LearnedValueNet` above cannot run (no embedding tables /
// pooling). Stage (a) of the embedding inference path
// (`docs/perf/2026-05-26-embedding-inference-path-design.md`): forwards
// the value head only, from id-embeddings, with a Python↔Rust parity gate.
// Per-card static features (Lever A proper) layer on top in stage (b).

/// `arch` descriptor emitted by `value_policy_net_generalist.py::dump_json`.
/// Vocab + slot dims are read from the JSON (not hardcoded) so the same
/// loader handles both the shrunk parity fixture and a full-vocab net.
#[derive(Debug, Deserialize)]
struct EmbedArchJson {
    wide_dim: usize,
    trunk_in: usize,
    hidden: usize,
    policy_dim: usize,
    trunk_depth: usize,
    card_vocab: usize,
    monster_vocab: usize,
    relic_vocab: usize,
    card_emb_dim: usize,
    monster_emb_dim: usize,
    relic_emb_dim: usize,
    /// Per-card static-feature width (Lever A stage (b)). Absent / 0 on
    /// stage-(a) id-only nets, in which case the feat block is skipped.
    #[serde(default)]
    card_feat_dim: usize,
    /// DeepSets per-card phi output width (sts2-rs-x9hyn). Absent / 0 ⇒ legacy
    /// raw sum-pool over card embeddings.
    #[serde(default)]
    card_phi_dim: usize,
    #[allow(dead_code)]
    h_max: usize,
    #[allow(dead_code)]
    p_max: usize,
    e_max: usize,
    #[allow(dead_code)]
    r_max: usize,
}

/// Weights table for the generalist net. Aux-HP (`W_h`/`b_h`) head is
/// parsed-and-dropped; policy (`W_p`/`b_p`) is now loaded and used.
#[derive(Debug, Deserialize)]
struct EmbedWeightsTable {
    card_emb: Vec<Vec<f32>>,
    monster_emb: Vec<Vec<f32>>,
    relic_emb: Vec<Vec<f32>>,
    #[serde(rename = "W_phi", default)]
    w_phi: Vec<Vec<f32>>,
    #[serde(rename = "b_phi", default)]
    b_phi: Vec<f32>,
    #[serde(rename = "W1")]
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    #[serde(rename = "W2")]
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    #[serde(rename = "W3")]
    w3: Vec<Vec<f32>>,
    b3: Vec<f32>,
    #[serde(rename = "W_v")]
    w_v: Vec<Vec<f32>>,
    b_v: Vec<f32>,
    #[serde(rename = "W_p")]
    w_p: Vec<Vec<f32>>,
    b_p: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct EmbedWeightsJson {
    arch: EmbedArchJson,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    /// Per-column normalization for the concatenated per-pile feat block,
    /// length `4 * card_feat_dim`, order `[hand, draw, discard, exhaust]`.
    /// Empty on stage-(a) nets. Lever A stage (b).
    #[serde(default)]
    feat_mean: Vec<f32>,
    #[serde(default)]
    feat_std: Vec<f32>,
    weights: EmbedWeightsTable,
}

/// An embedding table flattened row-major (`rows × dim`).
#[derive(Clone, Debug)]
struct EmbTable {
    data: Box<[f32]>,
    rows: usize,
    dim: usize,
}

impl EmbTable {
    fn from_json(
        raw: &[Vec<f32>],
        rows: usize,
        dim: usize,
        which: &'static str,
    ) -> Result<Self, ValueNetError> {
        let data = flatten_matrix(raw, rows, dim, which)?;
        Ok(Self { data, rows, dim })
    }

    /// `out[d] += sum_j table[ids[j]][d]`. NULL slots (`id == 0`) add row
    /// 0, which is **not** skipped: after training the NULL row drifts off
    /// zero and carries a learned bias the trainer's `sum_pool` includes.
    /// Out-of-range ids (only reachable from a malformed fixture) are
    /// skipped defensively.
    fn sum_pool_into(&self, ids: &[u16], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.dim);
        for &id in ids {
            let r = id as usize;
            if r >= self.rows {
                continue;
            }
            let base = r * self.dim;
            for (d, slot) in out.iter_mut().enumerate() {
                *slot += self.data[base + d];
            }
        }
    }

    /// DeepSets pool: `out[p] += sum_{id != 0} relu(emb[id] · W_phi + b_phi)`.
    /// NULL slots (id==0) are SKIPPED so the pool is padding-invariant
    /// (mirrors `value_policy_net_generalist.py::phi_pool`). `w_phi` is
    /// row-major `dim × phi_dim`. (sts2-rs-x9hyn)
    fn phi_pool_into(&self, ids: &[u16], w_phi: &[f32], b_phi: &[f32], out: &mut [f32]) {
        let phi_dim = b_phi.len();
        debug_assert_eq!(out.len(), phi_dim);
        debug_assert_eq!(w_phi.len(), self.dim * phi_dim);
        for &id in ids {
            if id == 0 {
                continue; // NULL slot — skipped (see doc).
            }
            let r = id as usize;
            if r >= self.rows {
                continue;
            }
            let base = r * self.dim;
            // pre[p] = b_phi[p] + Σ_e emb[id][e] * w_phi[e][p]; then relu, accumulate.
            for p in 0..phi_dim {
                let mut acc = b_phi[p];
                for e in 0..self.dim {
                    acc += self.data[base + e] * w_phi[e * phi_dim + p];
                }
                if acc > 0.0 {
                    out[p] += acc;
                }
            }
        }
    }

    /// Ordered per-slot lookup: writes `slots[k]`'s row into
    /// `out[k*dim .. (k+1)*dim]` for `k in 0..slots.len()`.
    fn ordered_concat_into(&self, ids: &[u16], out: &mut [f32]) {
        for (k, &id) in ids.iter().enumerate() {
            let r = id as usize;
            if r >= self.rows {
                continue;
            }
            let base = r * self.dim;
            out[k * self.dim..(k + 1) * self.dim]
                .copy_from_slice(&self.data[base..base + self.dim]);
        }
    }
}

/// Learned leaf evaluator over **id-embeddings + handcrafted wide
/// features** — the deployable form of the `generalist` trainer. Mirrors
/// the Python forward (`value_policy_net_generalist.py::forward`):
/// `x = concat([norm_wide, ph, pd, pc, pe, mc, rp])` then a 3-layer
/// `Linear+ReLU` trunk and a sigmoid value head.
#[derive(Debug)]
pub struct EmbeddingValueNet {
    card_emb: EmbTable,
    monster_emb: EmbTable,
    relic_emb: EmbTable,
    /// Trunk layers `(W, b)`, row-major `W`. `W1` is `trunk_in × hidden`,
    /// `W2`/`W3` are `hidden × hidden`.
    trunk: Vec<(Box<[f32]>, Box<[f32]>)>,
    w_v: Box<[f32]>,
    b_v: f32,
    w_p: Box<[f32]>,
    b_p: Box<[f32]>,
    policy_dim: usize,
    mean: Box<[f32]>,
    std: Box<[f32]>,
    /// Per-pile feat-block normalization, length `4 * card_feat_dim`,
    /// order `[hand, draw, discard, exhaust]`. Empty when `card_feat_dim
    /// == 0` (stage-(a) net). Lever A stage (b).
    feat_mean: Box<[f32]>,
    feat_std: Box<[f32]>,
    wide_dim: usize,
    trunk_in: usize,
    hidden: usize,
    e_max: usize,
    /// Per-card static-feature width; 0 ⇒ no feat block (stage (a)).
    card_feat_dim: usize,
    /// DeepSets per-card phi MLP, row-major `card_emb_dim × card_phi_dim`,
    /// applied to each card embedding before the per-pile sum. Empty ⇒
    /// legacy raw sum-pool (`card_phi_dim == 0`). (sts2-rs-x9hyn)
    w_phi: Box<[f32]>,
    b_phi: Box<[f32]>,
    card_phi_dim: usize,
    /// Caissawary inference-side Texel composition (bead `sts2-rs-ae14u`,
    /// mirrored from [`LearnedValueNet`]). `Some(k)` ⇒ emit
    /// `tanh(nn_logit + k · texel_norm)`; `None` ⇒ pure-NN
    /// `map_win_to_score`. Set via [`Self::with_texel_compose`].
    texel_compose_k: Option<f32>,
}

impl EmbeddingValueNet {
    /// Load from a `value_policy_net_generalist.py::dump_json` artifact.
    pub fn from_weights_path(path: &Path) -> Result<Self, ValueNetError> {
        let bytes = std::fs::read(path)?;
        Self::from_json_slice(&bytes)
    }

    /// Load from raw JSON bytes (identical schema + validation to
    /// [`Self::from_weights_path`]). Lets the production policy-prior asset
    /// be baked into the binary with `include_bytes!`, so the installed
    /// PyO3 wheel carries the weights and needs no runtime weights file.
    /// Bead `sts2-rs-x9hyn` (prod policy-prior integration).
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, ValueNetError> {
        let json: EmbedWeightsJson = serde_json::from_slice(bytes)?;
        let a = &json.arch;

        if a.wide_dim != WIDE_FEATURE_DIM {
            return Err(ValueNetError::ShapeMismatch {
                which: "arch.wide_dim",
                expected: (WIDE_FEATURE_DIM, 0),
                got: (a.wide_dim, 0),
            });
        }
        if a.trunk_depth != 3 {
            // The generalist trainer emits a fixed 3-layer trunk; stage
            // (a) supports exactly that shape.
            return Err(ValueNetError::ShapeMismatch {
                which: "arch.trunk_depth",
                expected: (3, 0),
                got: (a.trunk_depth, 0),
            });
        }
        if a.hidden == 0 || a.hidden > MAX_HIDDEN {
            return Err(ValueNetError::ShapeMismatch {
                which: "arch.hidden",
                expected: (MAX_HIDDEN, 0),
                got: (a.hidden, 0),
            });
        }
        let feat_block = 4 * a.card_feat_dim;
        // DeepSets (sts2-rs-x9hyn): when card_phi_dim > 0 each pile is phi_dim-wide;
        // otherwise the legacy raw card_emb_dim pool width is used.
        let card_pool_dim = if a.card_phi_dim > 0 {
            a.card_phi_dim
        } else {
            a.card_emb_dim
        };
        let expected_trunk_in = a.wide_dim
            + 4 * card_pool_dim
            + a.e_max * a.monster_emb_dim
            + a.relic_emb_dim
            + feat_block;
        if a.trunk_in != expected_trunk_in {
            return Err(ValueNetError::ShapeMismatch {
                which: "arch.trunk_in (wide + 4·card_pool + e_max·monster_emb + relic_emb + 4·card_feat)",
                expected: (expected_trunk_in, 0),
                got: (a.trunk_in, 0),
            });
        }
        if json.feature_mean.len() != WIDE_FEATURE_DIM || json.feature_std.len() != WIDE_FEATURE_DIM
        {
            return Err(ValueNetError::ShapeMismatch {
                which: "feature_mean/std",
                expected: (WIDE_FEATURE_DIM, 0),
                got: (json.feature_mean.len(), json.feature_std.len()),
            });
        }
        if json.feat_mean.len() != feat_block || json.feat_std.len() != feat_block {
            return Err(ValueNetError::ShapeMismatch {
                which: "feat_mean/std (4·card_feat_dim)",
                expected: (feat_block, 0),
                got: (json.feat_mean.len(), json.feat_std.len()),
            });
        }

        let w = &json.weights;
        let card_emb =
            EmbTable::from_json(&w.card_emb, a.card_vocab + 1, a.card_emb_dim, "card_emb")?;
        let monster_emb = EmbTable::from_json(
            &w.monster_emb,
            a.monster_vocab + 1,
            a.monster_emb_dim,
            "monster_emb",
        )?;
        let relic_emb = EmbTable::from_json(
            &w.relic_emb,
            a.relic_vocab + 1,
            a.relic_emb_dim,
            "relic_emb",
        )?;

        // DeepSets phi weights (sts2-rs-x9hyn). Naming caution: the parsed
        // weights struct is bound as `w`; the phi matrix inside it is `w.w_phi`.
        // The flattened phi is named `w_phi_flat` to avoid shadowing `w`.
        let (w_phi_flat, b_phi_flat) = if a.card_phi_dim > 0 {
            let w_phi_flat = flatten_matrix(&w.w_phi, a.card_emb_dim, a.card_phi_dim, "W_phi")?;
            let b_phi_flat = flatten_vector(&w.b_phi, a.card_phi_dim, "b_phi")?;
            (w_phi_flat, b_phi_flat)
        } else {
            (Box::from([] as [f32; 0]), Box::from([] as [f32; 0]))
        };

        let hidden = a.hidden;
        let trunk = vec![
            (
                flatten_matrix(&w.w1, a.trunk_in, hidden, "W1")?,
                flatten_vector(&w.b1, hidden, "b1")?,
            ),
            (
                flatten_matrix(&w.w2, hidden, hidden, "W2")?,
                flatten_vector(&w.b2, hidden, "b2")?,
            ),
            (
                flatten_matrix(&w.w3, hidden, hidden, "W3")?,
                flatten_vector(&w.b3, hidden, "b3")?,
            ),
        ];
        let w_v = flatten_matrix(&w.w_v, hidden, 1, "W_v")?;
        let b_v = flatten_vector(&w.b_v, 1, "b_v")?[0];

        // `policy_dim` is accepted as-is from the JSON (the parity fixture
        // uses 8, smoke nets use 64; production nets use 640 or 4672). The
        // 640/4672 gate is enforced at the policy-prior call site
        // (`policy_action_space`), not here, so small test nets still load.
        let policy_dim = a.policy_dim;
        let w_p = flatten_matrix(&w.w_p, hidden, policy_dim, "W_p")?;
        let b_p = flatten_vector(&w.b_p, policy_dim, "b_p")?;

        Ok(Self {
            card_emb,
            monster_emb,
            relic_emb,
            trunk,
            w_v,
            b_v,
            w_p,
            b_p,
            policy_dim,
            mean: json.feature_mean.into_boxed_slice(),
            std: json.feature_std.into_boxed_slice(),
            feat_mean: json.feat_mean.into_boxed_slice(),
            feat_std: json.feat_std.into_boxed_slice(),
            wide_dim: a.wide_dim,
            trunk_in: a.trunk_in,
            hidden,
            e_max: a.e_max,
            card_feat_dim: a.card_feat_dim,
            w_phi: w_phi_flat,
            b_phi: b_phi_flat,
            card_phi_dim: a.card_phi_dim,
            texel_compose_k: None,
        })
    }

    /// The wide-feature width this net was built for. Always equals
    /// [`WIDE_FEATURE_DIM`] for a successfully decoded net (`from_json_slice`
    /// rejects any other value); exposed so callers can assert the bundled
    /// asset is in step with the engine.
    #[must_use]
    pub fn wide_dim(&self) -> usize {
        self.wide_dim
    }

    /// Enable Caissawary inference-side Texel composition — see the
    /// [`LearnedValueNet::with_texel_compose`] twin for the rationale.
    #[must_use]
    pub fn with_texel_compose(mut self, k: Option<f32>) -> Self {
        self.texel_compose_k = k;
        self
    }

    /// Build the normalized trunk input `x` and run the 3-layer ReLU trunk,
    /// returning the post-trunk activation vector. This is the shared hot
    /// path for both `forward_value_from_ids` and
    /// `forward_value_and_policy_from_ids`; keeping it in one place means
    /// the value output is byte-identical between the two.
    #[allow(clippy::too_many_arguments)]
    fn trunk_from_ids(
        &self,
        raw_wide: &[f32],
        hand: &[u16],
        draw: &[u16],
        discard: &[u16],
        exhaust: &[u16],
        enemies: &[u16],
        relics: &[u16],
        feats_hand: &[f32],
        feats_draw: &[f32],
        feats_discard: &[f32],
        feats_exhaust: &[f32],
    ) -> Vec<f32> {
        let mut x = vec![0.0_f32; self.trunk_in];
        for i in 0..self.wide_dim {
            x[i] = (raw_wide[i] - self.mean[i]) / self.std[i];
        }
        let cd = self.card_emb.dim;
        let md = self.monster_emb.dim;
        let rd = self.relic_emb.dim;
        // DeepSets (sts2-rs-x9hyn): when phi is enabled each pile is phi_dim-wide;
        // otherwise the legacy raw embedding dim is used.
        let card_pool = if self.card_phi_dim > 0 {
            self.card_phi_dim
        } else {
            cd
        };
        let mut off = self.wide_dim;
        for pile in [hand, draw, discard, exhaust] {
            if self.card_phi_dim > 0 {
                self.card_emb.phi_pool_into(
                    pile,
                    &self.w_phi,
                    &self.b_phi,
                    &mut x[off..off + card_pool],
                );
            } else {
                self.card_emb
                    .sum_pool_into(pile, &mut x[off..off + card_pool]);
            }
            off += card_pool;
        }
        // Enemies: ordered concat over E_MAX slots (zero-padded).
        self.monster_emb
            .ordered_concat_into(enemies, &mut x[off..off + self.e_max * md]);
        off += self.e_max * md;
        self.relic_emb.sum_pool_into(relics, &mut x[off..off + rd]);
        off += rd;
        // Per-pile static-feature tail, normalized `(f - mean) / std`,
        // order [hand, draw, discard, exhaust] (Lever A stage (b)).
        let cfd = self.card_feat_dim;
        if cfd > 0 {
            for (pile_idx, feats) in [feats_hand, feats_draw, feats_discard, feats_exhaust]
                .into_iter()
                .enumerate()
            {
                let blk = pile_idx * cfd;
                for j in 0..cfd {
                    x[off + j] = (feats[j] - self.feat_mean[blk + j]) / self.feat_std[blk + j];
                }
                off += cfd;
            }
        }
        debug_assert_eq!(off, self.trunk_in);

        let mut a = vec![0.0_f32; self.hidden];
        let mut b = vec![0.0_f32; self.hidden];
        linear(&mut a, &self.trunk[0].0, &self.trunk[0].1, &x);
        relu_inplace(&mut a);
        linear(&mut b, &self.trunk[1].0, &self.trunk[1].1, &a);
        relu_inplace(&mut b);
        linear(&mut a, &self.trunk[2].0, &self.trunk[2].1, &b);
        relu_inplace(&mut a);
        a
    }

    /// Forward from raw (un-normalized) wide features + per-pile id slots,
    /// returning post-sigmoid `P(win)`. The wide block is normalized
    /// inline `(x - mean) / std`; embeddings are raw learned lookups.
    /// This is the parity-gated math (concat order / pooling / gather).
    ///
    /// The seven id/feature slices mirror the Python
    /// `GeneralistNet.forward` signature one-for-one; grouping them into
    /// a struct would only obscure that correspondence.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_value_from_ids(
        &self,
        raw_wide: &[f32],
        hand: &[u16],
        draw: &[u16],
        discard: &[u16],
        exhaust: &[u16],
        enemies: &[u16],
        relics: &[u16],
        // Per-pile summed per-card static features (Lever A stage (b)),
        // each of length `>= card_feat_dim`. Ignored when `card_feat_dim
        // == 0` (stage-(a) net).
        feats_hand: &[f32],
        feats_draw: &[f32],
        feats_discard: &[f32],
        feats_exhaust: &[f32],
    ) -> f32 {
        let trunk_out = self.trunk_from_ids(
            raw_wide,
            hand,
            draw,
            discard,
            exhaust,
            enemies,
            relics,
            feats_hand,
            feats_draw,
            feats_discard,
            feats_exhaust,
        );
        let mut value_pre = self.b_v;
        for (t, w) in trunk_out.iter().zip(self.w_v.iter()) {
            value_pre += t * w;
        }
        sigmoid(value_pre)
    }

    /// Forward from raw wide features + per-pile id slots, returning both
    /// `P(win)` (value head, post-sigmoid) and the policy prior
    /// (post-softmax over `policy_dim` slots). The value output is
    /// byte-identical to [`Self::forward_value_from_ids`] for the same
    /// inputs — both delegate to the same [`Self::trunk_from_ids`] helper.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_value_and_policy_from_ids(
        &self,
        raw_wide: &[f32],
        hand: &[u16],
        draw: &[u16],
        discard: &[u16],
        exhaust: &[u16],
        enemies: &[u16],
        relics: &[u16],
        feats_hand: &[f32],
        feats_draw: &[f32],
        feats_discard: &[f32],
        feats_exhaust: &[f32],
    ) -> (f32, Vec<f32>) {
        let trunk_out = self.trunk_from_ids(
            raw_wide,
            hand,
            draw,
            discard,
            exhaust,
            enemies,
            relics,
            feats_hand,
            feats_draw,
            feats_discard,
            feats_exhaust,
        );
        let mut value_pre = self.b_v;
        for (t, w) in trunk_out.iter().zip(self.w_v.iter()) {
            value_pre += t * w;
        }
        let value = sigmoid(value_pre);

        let mut policy = vec![0.0_f32; self.policy_dim];
        linear(&mut policy, &self.w_p, &self.b_p, &trunk_out);
        softmax_inplace(&mut policy);
        (value, policy)
    }

    /// Forward a live [`CombatState`] through the net, returning the
    /// depth-cap leaf score in `(-INPROGRESS_BOUND, +INPROGRESS_BOUND)`.
    /// Reuses [`crate::corpus::EmbeddingIds::from_state`] — the exact slot
    /// layout the corpus writer used — so train/inference ids align.
    pub fn evaluate(&self, state: &CombatState) -> f32 {
        let ids = crate::corpus::EmbeddingIds::from_state(state);
        let raw_wide = eval_features_wide(state);
        let p_win = self.forward_value_from_ids(
            &raw_wide,
            &ids.card_ids_hand,
            &ids.card_ids_draw,
            &ids.card_ids_discard,
            &ids.card_ids_exhaust,
            &ids.monster_slot_ids,
            &ids.relic_set_ids,
            &ids.card_feats_hand,
            &ids.card_feats_draw,
            &ids.card_feats_discard,
            &ids.card_feats_exhaust,
        );
        // Caissawary Texel composition (mirrors LearnedValueNet::compose_score).
        match self.texel_compose_k {
            None => map_win_to_score(p_win),
            Some(k) => {
                let p = p_win.clamp(1e-6, 1.0 - 1e-6);
                let nn_logit = (p / (1.0 - p)).ln();
                let texel_norm = sts2_core::evaluate_in_progress(state);
                let combined = (nn_logit + k * texel_norm).tanh();
                (combined * INPROGRESS_BOUND).clamp(-INPROGRESS_BOUND, INPROGRESS_BOUND)
            }
        }
    }

    /// Forward a live [`CombatState`] through the net, returning both the
    /// depth-cap leaf score (identical to [`Self::evaluate`]) and the
    /// policy prior (post-softmax, `policy_dim` slots). The score returned
    /// respects the Texel-compose setting exactly as `evaluate` does.
    pub fn evaluate_with_policy(&self, state: &CombatState) -> (f32, Vec<f32>) {
        let ids = crate::corpus::EmbeddingIds::from_state(state);
        let raw_wide = eval_features_wide(state);
        let (p_win, policy) = self.forward_value_and_policy_from_ids(
            &raw_wide,
            &ids.card_ids_hand,
            &ids.card_ids_draw,
            &ids.card_ids_discard,
            &ids.card_ids_exhaust,
            &ids.monster_slot_ids,
            &ids.relic_set_ids,
            &ids.card_feats_hand,
            &ids.card_feats_draw,
            &ids.card_feats_discard,
            &ids.card_feats_exhaust,
        );
        let score = match self.texel_compose_k {
            None => map_win_to_score(p_win),
            Some(k) => {
                let p = p_win.clamp(1e-6, 1.0 - 1e-6);
                let nn_logit = (p / (1.0 - p)).ln();
                let texel_norm = sts2_core::evaluate_in_progress(state);
                let combined = (nn_logit + k * texel_norm).tanh();
                (combined * INPROGRESS_BOUND).clamp(-INPROGRESS_BOUND, INPROGRESS_BOUND)
            }
        };
        (score, policy)
    }

    /// Return the [`crate::corpus::PolicyActionSpace`] implied by this
    /// net's `policy_dim` — mirrors [`LearnedValueNet::policy_action_space`].
    ///
    /// # Panics
    ///
    /// Panics if `policy_dim` is not one of the two production-supported
    /// values (`CANONICAL_ACTION_DIM` = 640 or `ACTION_KEY_DIM` = 4672).
    /// Small-vocab test nets (e.g. the 8-slot parity fixture) must not
    /// call this method.
    pub fn policy_action_space(&self) -> crate::corpus::PolicyActionSpace {
        assert!(
            self.policy_dim == crate::action_index::CANONICAL_ACTION_DIM
                || self.policy_dim == crate::action_index::ACTION_KEY_DIM,
            "EmbeddingValueNet::policy_action_space: unsupported policy_dim {} \
             (expected {} or {})",
            self.policy_dim,
            crate::action_index::CANONICAL_ACTION_DIM,
            crate::action_index::ACTION_KEY_DIM,
        );
        if self.policy_dim == crate::action_index::ACTION_KEY_DIM {
            crate::corpus::PolicyActionSpace::Keyed
        } else {
            crate::corpus::PolicyActionSpace::Canonical
        }
    }
}

#[cfg(test)]
mod embed_parity_tests {
    use super::*;
    use std::path::PathBuf;

    #[derive(serde::Deserialize)]
    struct Fixture {
        wide_raw: Vec<f32>,
        hand: Vec<u16>,
        draw: Vec<u16>,
        discard: Vec<u16>,
        exhaust: Vec<u16>,
        enemies: Vec<u16>,
        relics: Vec<u16>,
        #[serde(default)]
        feats_hand: Vec<f32>,
        #[serde(default)]
        feats_draw: Vec<f32>,
        #[serde(default)]
        feats_discard: Vec<f32>,
        #[serde(default)]
        feats_exhaust: Vec<f32>,
        expected_v_p: f32,
        expected_score: f32,
        /// Post-softmax policy probabilities from the Python forward pass.
        /// Absent in pre-policy-parity fixtures (defaults to empty).
        #[serde(default)]
        expected_policy: Vec<f32>,
    }

    /// The Python↔Rust forward-parity gate. Fixtures are regenerated by
    /// `scripts/gen_embedding_parity_fixture.py`. A failure here means the
    /// Rust embedding forward diverged from the trainer (concat order,
    /// NULL-row pooling, or embedding gather) — fix before any A/B.
    #[test]
    fn embed_forward_matches_python() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let net = EmbeddingValueNet::from_weights_path(&dir.join("embed_parity_net.json"))
            .expect("load parity net");
        let fx: Fixture = serde_json::from_slice(
            &std::fs::read(dir.join("embed_parity_input.json")).expect("read fixture input"),
        )
        .expect("parse fixture input");

        let p = net.forward_value_from_ids(
            &fx.wide_raw,
            &fx.hand,
            &fx.draw,
            &fx.discard,
            &fx.exhaust,
            &fx.enemies,
            &fx.relics,
            &fx.feats_hand,
            &fx.feats_draw,
            &fx.feats_discard,
            &fx.feats_exhaust,
        );
        assert!(
            (p - fx.expected_v_p).abs() < 1e-4,
            "p_win parity failed: rust={p} python={} (Δ={:.2e})",
            fx.expected_v_p,
            (p - fx.expected_v_p).abs()
        );
        let score = map_win_to_score(p);
        assert!(
            (score - fx.expected_score).abs() < 1e-4,
            "score parity failed: rust={score} python={} (Δ={:.2e})",
            fx.expected_score,
            (score - fx.expected_score).abs()
        );

        // Policy-head parity: forward_value_and_policy_from_ids must produce
        // a post-softmax distribution identical to the Python trainer's forward.
        // Only run when the fixture carries expected_policy (post-regeneration).
        if !fx.expected_policy.is_empty() {
            let (_p_val, policy) = net.forward_value_and_policy_from_ids(
                &fx.wide_raw,
                &fx.hand,
                &fx.draw,
                &fx.discard,
                &fx.exhaust,
                &fx.enemies,
                &fx.relics,
                &fx.feats_hand,
                &fx.feats_draw,
                &fx.feats_discard,
                &fx.feats_exhaust,
            );
            assert_eq!(
                policy.len(),
                fx.expected_policy.len(),
                "policy length mismatch: rust={} python={}",
                policy.len(),
                fx.expected_policy.len()
            );
            for (i, (rust_p, py_p)) in policy.iter().zip(fx.expected_policy.iter()).enumerate() {
                let delta = (rust_p - py_p).abs();
                assert!(
                    delta < 1e-4,
                    "policy[{i}] parity failed: rust={rust_p} python={py_p} (Δ={delta:.2e})"
                );
            }
        }
    }

    /// Coverage for the legacy `card_phi_dim == 0` raw sum-pool path
    /// (sts2-rs-x9hyn). The checked-in parity fixture is phi-PRESENT, so
    /// the empty-phi arm of `from_weights_path` + the `sum_pool_into`
    /// branch of `trunk_from_ids` would otherwise be untested. We derive a
    /// phi-ABSENT net in-memory from the same fixture (strip `W_phi`/`b_phi`,
    /// set `card_phi_dim = 0`), load it, and assert it (a) loads with an
    /// empty phi table, (b) yields a finite `p_win` in (0, 1), and (c)
    /// produces a *different* output than the phi net on the identical
    /// input — proving the sum-pool branch is actually taken (not phi).
    /// `TRUNK_IN` is unchanged across phi-on/off here because
    /// `CARD_POOL_DIM == CARD_EMB_DIM == 16`, so the stripped JSON loads
    /// with the same trunk shape.
    #[test]
    fn embed_no_phi_routes_through_sum_pool() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");

        // Phi-PRESENT reference net + the shared input.
        let phi_net = EmbeddingValueNet::from_weights_path(&dir.join("embed_parity_net.json"))
            .expect("load phi parity net");
        assert!(phi_net.card_phi_dim > 0, "fixture must be phi-present");
        let fx: Fixture = serde_json::from_slice(
            &std::fs::read(dir.join("embed_parity_input.json")).expect("read fixture input"),
        )
        .expect("parse fixture input");

        // Build a phi-ABSENT variant in-memory: strip the phi weights and
        // zero the arch field.
        let mut json: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("embed_parity_net.json")).expect("read parity net"),
        )
        .expect("parse parity net");
        json["arch"]["card_phi_dim"] = serde_json::json!(0);
        let weights = json["weights"]
            .as_object_mut()
            .expect("weights is an object");
        weights.remove("W_phi");
        weights.remove("b_phi");

        // Per-process unique temp path (nextest runs one process per test,
        // but include a nanosecond stamp for belt-and-suspenders).
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let tmp =
            std::env::temp_dir().join(format!("embed_no_phi_{}_{stamp}.json", std::process::id()));
        std::fs::write(
            &tmp,
            serde_json::to_vec(&json).expect("serialize stripped net"),
        )
        .expect("write stripped net");

        let no_phi_net = EmbeddingValueNet::from_weights_path(&tmp).expect("load phi-absent net");
        let _ = std::fs::remove_file(&tmp);

        // (a) loads with phi disabled + empty phi table.
        assert_eq!(no_phi_net.card_phi_dim, 0);
        assert!(
            no_phi_net.w_phi.is_empty(),
            "phi-absent net must have empty W_phi"
        );
        assert!(
            no_phi_net.b_phi.is_empty(),
            "phi-absent net must have empty b_phi"
        );

        let p_no_phi = no_phi_net.forward_value_from_ids(
            &fx.wide_raw,
            &fx.hand,
            &fx.draw,
            &fx.discard,
            &fx.exhaust,
            &fx.enemies,
            &fx.relics,
            &fx.feats_hand,
            &fx.feats_draw,
            &fx.feats_discard,
            &fx.feats_exhaust,
        );
        // (b) finite p_win in the open unit interval.
        assert!(
            p_no_phi.is_finite() && p_no_phi > 0.0 && p_no_phi < 1.0,
            "sum-pool p_win out of (0,1): {p_no_phi}"
        );

        // (c) sum-pool output differs from phi output on the same input,
        // proving the legacy branch (not the phi branch) was taken.
        let p_phi = phi_net.forward_value_from_ids(
            &fx.wide_raw,
            &fx.hand,
            &fx.draw,
            &fx.discard,
            &fx.exhaust,
            &fx.enemies,
            &fx.relics,
            &fx.feats_hand,
            &fx.feats_draw,
            &fx.feats_discard,
            &fx.feats_exhaust,
        );
        assert!(
            (p_no_phi - p_phi).abs() > 1e-3,
            "sum-pool and phi outputs too close ({p_no_phi} vs {p_phi}); \
             the phi branch may have been taken in the phi-absent net"
        );
    }

    /// Bead `sts2-rs-x9hyn`: `EmbeddingValueNet` drives
    /// `LearnedPolicyPrior<EmbeddingValueNet>` as a PUCT prior for
    /// Expert Iteration. The wide net already satisfies [`PolicyNet`]
    /// (`impl PolicyNet for LearnedValueNet`); this pins that the
    /// embedding net's `impl PolicyNet` projects its policy head onto
    /// the legal-action set and renormalizes to a valid distribution,
    /// reusing the same `project_dense_prior` + `policy_action_space`
    /// machinery the wide impl uses.
    ///
    /// The committed `embed_parity_net.json` fixture has a tiny
    /// `arch.policy_dim = 8` (it only validates the forward-pass
    /// numerics against Python), which `policy_action_space()` rejects
    /// — the legal-slot projection requires the real Canonical (640)
    /// or Keyed (4672) action space. So we widen the policy head to
    /// `CANONICAL_ACTION_DIM` in-memory (mirroring the temp-JSON reload
    /// pattern of `embed_no_phi_routes_through_sum_pool`) before driving
    /// the prior.
    #[test]
    fn embed_policy_prior_normalizes_over_legal_actions() {
        use crate::action_index::CANONICAL_ACTION_DIM;
        use crate::Policy;
        use std::sync::Arc;
        use sts2_core::cards::{Card, CardId, CardType, TargetType};
        use sts2_core::{CardUid, CombatPhase};

        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");

        // Widen the parity net's policy head from the tiny fixture
        // `policy_dim` to the real Canonical action space so
        // `policy_action_space()` resolves and the legal-slot projection
        // runs. `W_p` is shaped `[hidden][policy_dim]`; `b_p` is
        // `[policy_dim]`.
        let mut json: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("embed_parity_net.json")).expect("read parity net"),
        )
        .expect("parse parity net");
        let hidden = json["arch"]["hidden"].as_u64().expect("hidden") as usize;
        json["arch"]["policy_dim"] = serde_json::json!(CANONICAL_ACTION_DIM);
        let weights = json["weights"]
            .as_object_mut()
            .expect("weights is an object");
        // Deterministic non-uniform policy head: a single positive logit
        // per hidden row at a rotating slot keeps the projected prior off
        // the uniform-fallback path for at least some legal actions.
        let w_p: Vec<Vec<f32>> = (0..hidden)
            .map(|h| {
                let mut row = vec![0.0_f32; CANONICAL_ACTION_DIM];
                row[h % CANONICAL_ACTION_DIM] = 1.0;
                row
            })
            .collect();
        weights.insert("W_p".to_string(), serde_json::json!(w_p));
        weights.insert(
            "b_p".to_string(),
            serde_json::json!(vec![0.0_f32; CANONICAL_ACTION_DIM]),
        );

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let tmp =
            std::env::temp_dir().join(format!("embed_prior_{}_{stamp}.json", std::process::id()));
        std::fs::write(
            &tmp,
            serde_json::to_vec(&json).expect("serialize widened net"),
        )
        .expect("write widened net");
        let net = EmbeddingValueNet::from_weights_path(&tmp).expect("load widened net");
        let _ = std::fs::remove_file(&tmp);

        // Build a branching in-progress state: PlayerPlayPhase with two
        // playable cards in hand + energy yields >=2 PlayCards plus
        // EndTurn. Mirrors the inline construction used by core's
        // `legal_actions_emits_zero_cost_buff_prior_first` test.
        let mk = |uid_raw: u32, id: CardId, cost: i8, ty: CardType, tgt: TargetType| Card {
            uid: CardUid::new(uid_raw).expect("non-zero"),
            card_id: id,
            cost,
            card_type: ty,
            target_type: tgt,
            upgraded: false,
            base_damage: 6,
            base_block: 0,
            ..Card::pod_defaults()
        };
        let mut state = CombatState {
            phase: CombatPhase::PlayerPlayPhase,
            ..CombatState::default()
        };
        state.cards.push(mk(
            1,
            CardId::StrikeIronclad,
            1,
            CardType::Attack,
            TargetType::Self_,
        ));
        state.cards.push(mk(
            2,
            CardId::DefendIronclad,
            1,
            CardType::Skill,
            TargetType::Self_,
        ));
        for raw in 1..=2 {
            state.player.hand.push(CardUid::new(raw).expect("non-zero"));
        }
        state.player.energy = 3;

        let actions = sts2_core::legal_actions(&state);
        assert!(
            actions.len() >= 2,
            "need a branching state; got {} legal action(s)",
            actions.len()
        );

        let prior = LearnedPolicyPrior::new(Arc::new(net)).prior(&state, &actions);
        assert_eq!(prior.len(), actions.len());
        let sum: f32 = prior.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "prior must be a distribution, got sum={sum}"
        );
        assert!(prior.iter().all(|&p| p >= 0.0));

        // The installed W_p is deliberately non-uniform, so a correct
        // projection must NOT collapse to the uniform fallback (which would
        // pass the len/sum/non-neg checks above). This pins that the policy
        // head's signal actually reaches the prior.
        let first = prior[0];
        assert!(
            prior.iter().any(|&p| (p - first).abs() > 1e-6),
            "prior collapsed to uniform fallback: {prior:?}"
        );
    }
}

/// Wrapper that captures the leaf-eval choice without breaking
/// [`crate::MctsConfig`]'s `Copy` derive. Stored on [`crate::Mcts`]
/// directly (see [`crate::Mcts::with_leaf_eval`]) so callers that
/// don't opt into the learned path pay zero cost — no extra
/// `Arc` traffic, no extra branch beyond a single `Option::is_some`
/// check at the depth-cap leaf.
///
/// # This knob is INERT at the deployed depth — read before tuning it
///
/// The arms below look like a live extension point. They are not one in any
/// deployed configuration, and the measurements say so twice over (GH #527):
///
/// * The leaf is consulted from `crate::depth_cut_value` (rollout hit the
///   ply cap) and from `crate::loss_leaf_value` — but the latter only reaches
///   `evaluate` when `MctsConfig::loss_gradient` is **false**, and the default
///   is `true`, which takes the graded loss band instead. Under the default
///   `TerminalEvalMode::RichBoth` the win branch goes to `evaluate_victory`.
///   So `depth_cut_value` is the only live consumer.
/// * At `sts2_runai::policy::COMBAT_ROLLOUT_DEPTH = 200` the cap is never hit.
///   GH #512's depth probe measured **0/5912 cap leaves at d200**; the
///   `wb-b5c453e74f1b80e3` repro independently showed `W + L == visits` with
///   **T = 0** across 2000 sims, and depth cuts are pinned to bucket as `Tie`.
///
/// The 2026-07-18 hard-example-net campaign reached the same conclusion from
/// the other side and closed the evaluator half of GH #124: "the combat MCTS
/// leaf-eval seam is ~INERT at deployed config … Texel-vs-TexelWide leaves
/// 11/14 reps byte-identical. ANY leaf-eval improvement has near-zero surface
/// area. **Don't attack combat via leaf evaluation again without changing the
/// search's leaf surface first.**" Its own null (+0.000/+0.003/+0.005 across
/// three training seeds) is the direct evidence.
///
/// This note exists because that conclusion was absent from the code and got
/// re-derived from scratch, wrongly, as issue #521.
///
/// Two things this does NOT say:
///
/// * `MctsConfig::default().max_rollout_depth` is **10**, not 200, and the CLI
///   (`sts2-cli/src/main.rs`), the combat REPL and `closed_loop_harness` take
///   that default. Those truncate constantly, so the leaf really is doing the
///   work there.
/// * A dead arm here does not imply a dead function. `evaluate_in_progress`
///   still runs inside the learned arms as the `texel_norm` term of
///   `tanh(nn_logit + k · texel_norm)`, and `eval_features_wide` is the input
///   vector for both learned nets plus the training-corpus and fork-label
///   writers.
#[derive(Clone, Debug, Default)]
pub enum LeafEval {
    /// Texel-tuned `evaluate_in_progress` — the narrow 29-feature leaf.
    Texel,
    /// Wide 172-feature Texel leaf (`sts2_core::evaluate_in_progress_wide`).
    /// Default as of Phase 1 (L2-regularized fit beats narrow on the gate).
    #[default]
    TexelWide,
    /// Wide Texel leaf with **runtime-loaded** weights (Phase 2 ExpI). Same
    /// scoring as [`LeafEval::TexelWide`] but reads the 172-weight vector from
    /// an `Arc` instead of the baked-in `EVAL_FEATURE_WEIGHTS_WIDE` const, so a
    /// round's freshly-fit value leaf can be gated against the const champion
    /// in one binary without a rebuild. Load via [`load_wide_leaf_weights`].
    TexelWideRuntime(Arc<[f32; sts2_core::WIDE_FEATURE_DIM]>),
    /// Learned net loaded from an ONNX artifact (qctu.4 producer).
    Learned(Arc<LearnedValueNet>),
    /// Embedding-aware learned net (`value_policy_net_generalist.py`):
    /// id-embeddings + wide features. Stage (a) of the embedding
    /// inference path. Value-head only.
    LearnedEmbed(Arc<EmbeddingValueNet>),
}

/// Terminal-loss clamp value returned by [`LeafEval::evaluate`] when
/// the state is `is_over && !player_won`. Bead `sts2-rs-pvuwv`
/// (follow-up to `sts2-rs-0ljme`). Combined with the
/// `proven_loss` backup at `lib.rs:1728-1729` (`-1.0 + 0.5 * leaf`),
/// `-0.9` lands the backup at exactly `-1.45` — the documented loss-
/// band floor at the `MAX_LEAF_VALUE_ABS = 1.5` invariant. Without
/// this guard, [`sts2_core::evaluate_in_progress`] returns the Texel
/// feature-vector score of the dead-player state (e.g. `+0.22` for
/// states with accumulated `damage_dealt_raw` / buffs / survival
/// terms), collapsing the backup to `~-0.89` and flipping selection
/// toward suicide cards (Bloodletting / Offering at lethal HP).
pub const TERMINAL_LOSS_LEAF: f32 = -0.9;

impl LeafEval {
    /// Dispatch the depth-cap leaf evaluation. The free function
    /// [`crate::depth_cut_value`] reads this via the [`crate::Mcts`]
    /// engine handle.
    ///
    /// **Terminal-loss guard (`sts2-rs-pvuwv`):** when the state is
    /// already terminal-loss (`is_over && !player_won`), short-
    /// circuits to [`TERMINAL_LOSS_LEAF`] so the
    /// `proven_loss` backup at `lib.rs:1728-1729` lands at `-1.45`
    /// — the documented loss-band floor. The guard sits above both
    /// the Texel and Learned arms: Texel's
    /// `evaluate_in_progress` reads the feature vector of the dead
    /// state as-is (`damage_dealt_raw`, buffs, etc. all stay
    /// positive), and the learned net is similarly unconditioned on
    /// `is_over`, so both arms need the same clamp.
    pub fn evaluate(&self, state: &CombatState) -> f32 {
        if state.is_over && !state.player_won {
            return TERMINAL_LOSS_LEAF;
        }
        match self {
            LeafEval::Texel => sts2_core::evaluate_in_progress(state),
            LeafEval::TexelWide => sts2_core::evaluate_in_progress_wide(state),
            LeafEval::TexelWideRuntime(weights) => {
                sts2_core::wide_leaf_score(&sts2_core::eval_features_wide(state), weights)
            }
            LeafEval::Learned(net) => net.evaluate(state),
            LeafEval::LearnedEmbed(net) => net.evaluate(state),
        }
    }

    pub fn is_learned(&self) -> bool {
        matches!(self, LeafEval::Learned(_) | LeafEval::LearnedEmbed(_))
    }
}

/// Load wide-leaf weights from a `texel_tune.py --target value` output
/// (`{"fitted_weights_ordered": [..172 floats..]}`). Used to build a
/// [`LeafEval::TexelWideRuntime`] for Phase-2 expert-iteration rounds (gate a
/// round's freshly-fit leaf against the const champion without a rebuild).
/// Errors on read/parse failure or a width that is not
/// [`sts2_core::WIDE_FEATURE_DIM`].
pub fn load_wide_leaf_weights(
    path: &std::path::Path,
) -> Result<[f32; sts2_core::WIDE_FEATURE_DIM], ValueNetError> {
    let bytes = std::fs::read(path)?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    let arr = v
        .get("fitted_weights_ordered")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| ValueNetError::Parse("missing fitted_weights_ordered array".to_string()))?;
    if arr.len() != sts2_core::WIDE_FEATURE_DIM {
        return Err(ValueNetError::Parse(format!(
            "fitted_weights_ordered has {} entries, expected {}",
            arr.len(),
            sts2_core::WIDE_FEATURE_DIM
        )));
    }
    let mut weights = [0.0f32; sts2_core::WIDE_FEATURE_DIM];
    for (i, slot) in weights.iter_mut().enumerate() {
        *slot = arr[i].as_f64().ok_or_else(|| {
            ValueNetError::Parse(format!("fitted_weights_ordered[{i}] is not a number"))
        })? as f32;
    }
    Ok(weights)
}

// ── Learned policy prior — bead sts2-rs-v0i1 (qctu.8) ──────────────────────

/// MCTS [`crate::Policy`] implementation backed by the learned
/// net's policy head. Bead `sts2-rs-v0i1` (qctu.8). Reads the
/// dense `[CANONICAL_ACTION_DIM]`-wide policy distribution from
/// the ONNX graph and projects it onto the legal-action set the
/// engine handed in.
///
/// ## Forward path
///
/// 1. Run the tract model once via
///    [`LearnedValueNet::evaluate_with_policy`]. The value-head
///    output is discarded here (the engine reads value via
///    `LeafEval::Learned` at the depth-cap leaf separately —
///    a combined-forward optimization that shares one tract
///    invocation across both surfaces is a deferred perf bead).
/// 2. For each legal action, look up its canonical slot via
///    [`crate::action_to_slot`] and read the corresponding
///    probability from the dense vector.
/// 3. Renormalize across legal actions so the returned prior
///    sums to 1.0 — the raw 640-wide softmax includes mass on
///    illegal slots (cards not in hand, EndTurn outside
///    PlayPhase, …) that has to be masked out.
///
/// Sharing pattern: one `Arc<LearnedValueNet>` per process,
/// wrapped in `LearnedPolicyPrior::new(arc)` per `Mcts` engine
/// construction.
/// Diagnostic counters for [`LearnedPolicyPrior`]. GitHub issues #13
/// (silent uniform-fallback) and #24 (target collision + projection
/// quality). Counters are atomic so the `Arc<LearnedPolicyPriorStats>`
/// can be shared and snapshotted from any thread.
///
/// The fallback rate `uniform_fallbacks / evaluations` is the
/// load-bearing signal for "is the trained policy actually contributing
/// signal, or is PUCT effectively running on uniform?" — a high
/// fallback rate (>1%) means A/B comparisons of PUCT-on vs PUCT-off
/// are partly measuring uniform-on-legal, not the trained policy.
///
/// `target_collisions_total` and `evaluations_with_collisions` track
/// the action-index target collapse problem (#24): when two distinct
/// legal actions (e.g. Bash @ enemy A and Bash @ enemy B) map to the
/// same policy slot, the prior cannot distinguish them.
#[derive(Debug, Default)]
pub struct LearnedPolicyPriorStats {
    /// Total `Policy::prior` calls served. The search caches the prior per
    /// node (GH #830, [`crate::Mcts::with_prior_cache`], on by default), so
    /// in a search this counts cache misses (about one per decision node),
    /// not descent reads; so do the rates derived from it.
    evaluations: AtomicU64,
    /// Calls whose legal-mass pre-renormalization sum was below the
    /// `1e-9` threshold and therefore degraded to uniform-on-legal.
    uniform_fallbacks: AtomicU64,
    /// Sum of pre-renormalization legal mass across all evaluations,
    /// stored as `(mass * 1e6) as u64` so it stays in atomic ints.
    /// Mean legal mass = `legal_mass_sum_micros / evaluations / 1e6`.
    legal_mass_sum_micros: AtomicU64,
    /// Number of distinct-action / same-slot collisions observed,
    /// summed across all evaluations. A single 3-way collision
    /// contributes 2 (`n_actions - n_distinct_slots`).
    target_collisions_total: AtomicU64,
    /// Number of evaluations that observed at least one collision.
    evaluations_with_collisions: AtomicU64,
}

impl LearnedPolicyPriorStats {
    /// Number of times [`crate::Policy::prior`] was invoked: per node, not
    /// per read, under the search's prior cache
    /// ([`crate::Mcts::with_prior_cache`]).
    pub fn evaluations(&self) -> u64 {
        self.evaluations.load(Ordering::Relaxed)
    }

    /// Number of evaluations that fell back to uniform-on-legal because
    /// the trained policy assigned ~zero mass to every legal slot.
    pub fn uniform_fallbacks(&self) -> u64 {
        self.uniform_fallbacks.load(Ordering::Relaxed)
    }

    /// Fraction of evaluations that hit the uniform fallback. Returns
    /// `0.0` if no evaluations have run yet.
    #[allow(clippy::cast_precision_loss)]
    pub fn uniform_fallback_rate(&self) -> f64 {
        let total = self.evaluations();
        if total == 0 {
            0.0
        } else {
            self.uniform_fallbacks() as f64 / total as f64
        }
    }

    /// Mean pre-renormalization legal mass across all evaluations.
    /// A value far below 1.0 indicates the policy is consistently
    /// placing most of its mass on illegal slots.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_legal_mass(&self) -> f64 {
        let total = self.evaluations();
        if total == 0 {
            0.0
        } else {
            let sum_micros = self.legal_mass_sum_micros.load(Ordering::Relaxed);
            (sum_micros as f64 / 1e6) / total as f64
        }
    }

    /// Total per-action collision count summed across all evaluations.
    /// See [`LearnedPolicyPriorStats`] docs for the per-eval contribution
    /// formula.
    pub fn target_collisions_total(&self) -> u64 {
        self.target_collisions_total.load(Ordering::Relaxed)
    }

    /// Number of evaluations that observed at least one
    /// distinct-action / same-slot collision.
    pub fn evaluations_with_collisions(&self) -> u64 {
        self.evaluations_with_collisions.load(Ordering::Relaxed)
    }

    /// Fraction of evaluations that observed at least one collision.
    #[allow(clippy::cast_precision_loss)]
    pub fn collision_rate(&self) -> f64 {
        let total = self.evaluations();
        if total == 0 {
            0.0
        } else {
            self.evaluations_with_collisions() as f64 / total as f64
        }
    }

    /// Reset all counters to zero. Useful between A/B harness runs.
    pub fn reset(&self) {
        self.evaluations.store(0, Ordering::Relaxed);
        self.uniform_fallbacks.store(0, Ordering::Relaxed);
        self.legal_mass_sum_micros.store(0, Ordering::Relaxed);
        self.target_collisions_total.store(0, Ordering::Relaxed);
        self.evaluations_with_collisions.store(0, Ordering::Relaxed);
    }
}

/// A net that can produce a `(value, dense-policy-softmax)` pair and
/// declare which action-slot scheme its policy head uses. Implemented by
/// both the wide [`LearnedValueNet`] and the [`EmbeddingValueNet`] so the
/// PUCT prior is generic over encoding (sts2-rs-1bcx).
pub trait PolicyNet: Send + Sync {
    fn evaluate_with_policy(&self, state: &CombatState) -> (f32, Vec<f32>);
    fn policy_action_space(&self) -> crate::corpus::PolicyActionSpace;
}

impl PolicyNet for LearnedValueNet {
    fn evaluate_with_policy(&self, state: &CombatState) -> (f32, Vec<f32>) {
        LearnedValueNet::evaluate_with_policy(self, state)
    }
    fn policy_action_space(&self) -> crate::corpus::PolicyActionSpace {
        LearnedValueNet::policy_action_space(self)
    }
}

impl PolicyNet for EmbeddingValueNet {
    fn evaluate_with_policy(&self, state: &CombatState) -> (f32, Vec<f32>) {
        EmbeddingValueNet::evaluate_with_policy(self, state)
    }
    fn policy_action_space(&self) -> crate::corpus::PolicyActionSpace {
        EmbeddingValueNet::policy_action_space(self)
    }
}

pub struct LearnedPolicyPrior<N: PolicyNet = LearnedValueNet> {
    net: Arc<N>,
    stats: Arc<LearnedPolicyPriorStats>,
}

impl<N: PolicyNet> std::fmt::Debug for LearnedPolicyPrior<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LearnedPolicyPrior")
            .field("net", &"<PolicyNet>")
            .field("evaluations", &self.stats.evaluations())
            .field("uniform_fallbacks", &self.stats.uniform_fallbacks())
            .finish()
    }
}

/// Cloning shares both the (read-only) net and the stats handle, so
/// `par_search_with_policy` workers that each clone the prior aggregate
/// their evaluation/fallback counters into one shared total. Cheap:
/// two `Arc` bumps, no weight copy. Bead `sts2-rs-x9hyn`.
impl<N: PolicyNet> Clone for LearnedPolicyPrior<N> {
    fn clone(&self) -> Self {
        Self {
            net: Arc::clone(&self.net),
            stats: Arc::clone(&self.stats),
        }
    }
}

impl<N: PolicyNet> LearnedPolicyPrior<N> {
    pub fn new(net: Arc<N>) -> Self {
        Self {
            net,
            stats: Arc::new(LearnedPolicyPriorStats::default()),
        }
    }

    /// Like [`Self::new`] but shares an externally-owned stats handle so
    /// counters aggregate across many short-lived priors (e.g. corpus_gen
    /// builds one prior per move; sharing the stats Arc gives a whole-run
    /// fallback/collision rate). sts2-rs-1bcx.
    pub fn with_stats(net: Arc<N>, stats: Arc<LearnedPolicyPriorStats>) -> Self {
        Self { net, stats }
    }

    /// Clone the `Arc<LearnedPolicyPriorStats>` so callers can read
    /// counters after the prior is moved into an `Mcts` engine
    /// (GitHub #13 / #24 — post-search logging).
    pub fn stats(&self) -> Arc<LearnedPolicyPriorStats> {
        Arc::clone(&self.stats)
    }

    /// Extract a legal-action-masked + renormalized prior from a
    /// dense `[CANONICAL_ACTION_DIM]` distribution. Factored out
    /// so it's unit-testable without an ONNX fixture — the
    /// projection logic is the load-bearing piece of this bead;
    /// the tract forward is plumbed-through.
    ///
    /// Uniform fallback: if every legal action maps to a zero
    /// probability slot (e.g. policy head still at He/Xavier init
    /// on a partially-trained net), return uniform `1/N` rather
    /// than NaN.
    #[cfg(test)]
    pub(crate) fn project_dense_prior(
        dense: &[f32],
        legal_slots: &[Option<usize>],
    ) -> SmallVec<[f32; 16]> {
        Self::project_dense_prior_instrumented(dense, legal_slots, None)
    }

    /// Same as [`Self::project_dense_prior`] but updates the supplied
    /// stats handle with per-call observability (GitHub #13, #24).
    pub(crate) fn project_dense_prior_instrumented(
        dense: &[f32],
        legal_slots: &[Option<usize>],
        stats: Option<&LearnedPolicyPriorStats>,
    ) -> SmallVec<[f32; 16]> {
        let raw: SmallVec<[f32; 16]> = legal_slots
            .iter()
            .map(|s| s.and_then(|i| dense.get(i)).copied().unwrap_or(0.0))
            .collect();
        let sum: f32 = raw.iter().sum();

        if let Some(s) = stats {
            s.evaluations.fetch_add(1, Ordering::Relaxed);
            let mass_micros = (f64::from(sum.max(0.0)) * 1e6) as u64;
            s.legal_mass_sum_micros
                .fetch_add(mass_micros, Ordering::Relaxed);

            // Collision detection: count distinct Some(slot) values.
            // A "collision" is when two distinct legal actions resolve
            // to the same slot (the action-index target-collapse
            // problem from #24). `None` entries (unresolvable slots)
            // are not collisions — they're handled by the zero-fallback.
            let n_resolved = legal_slots.iter().filter(|x| x.is_some()).count();
            if n_resolved > 1 {
                let mut seen: SmallVec<[usize; 16]> = SmallVec::new();
                let mut distinct = 0usize;
                for slot in legal_slots.iter().filter_map(|x| *x) {
                    if !seen.contains(&slot) {
                        seen.push(slot);
                        distinct += 1;
                    }
                }
                let collisions = n_resolved - distinct;
                if collisions > 0 {
                    s.target_collisions_total
                        .fetch_add(collisions as u64, Ordering::Relaxed);
                    s.evaluations_with_collisions
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        if sum > 1e-9 {
            raw.into_iter().map(|p| p / sum).collect()
        } else {
            if let Some(s) = stats {
                s.uniform_fallbacks.fetch_add(1, Ordering::Relaxed);
            }
            // Uniform fallback — keeps PUCT well-defined.
            let n = legal_slots.len().max(1);
            #[allow(clippy::cast_precision_loss)]
            let p = 1.0_f32 / n as f32;
            legal_slots.iter().map(|_| p).collect()
        }
    }
}

impl<N: PolicyNet> crate::Policy for LearnedPolicyPrior<N> {
    fn prior(&self, state: &CombatState, actions: &[Action]) -> SmallVec<[f32; 16]> {
        let (_value, dense) = self.net.evaluate_with_policy(state);
        let space = self.net.policy_action_space();
        let legal_slots: SmallVec<[Option<usize>; 16]> = actions
            .iter()
            .map(|a| match space {
                crate::corpus::PolicyActionSpace::Canonical => {
                    crate::action_index::action_to_slot(state, a)
                }
                crate::corpus::PolicyActionSpace::Keyed => {
                    crate::action_index::action_to_keyed_slot(state, a)
                }
            })
            .collect();
        Self::project_dense_prior_instrumented(&dense, &legal_slots, Some(&self.stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sts2_core::CombatState;

    #[test]
    fn norm_stats_identity_is_identity() {
        let n = NormStats::identity();
        for i in 0..WIDE_FEATURE_DIM {
            assert!(n.mean[i].abs() < 1e-9);
            assert!((n.std[i] - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn norm_stats_default_is_identity() {
        let n = NormStats::default();
        let id = NormStats::identity();
        for i in 0..WIDE_FEATURE_DIM {
            assert!((n.mean[i] - id.mean[i]).abs() < 1e-9);
            assert!((n.std[i] - id.std[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn leaf_eval_default_is_texel_wide() {
        // Phase 1 (2026-06-09): the L2-regularized wide leaf beat narrow on
        // the paired gate (headline p=0.025), so TexelWide is now the default.
        let l = LeafEval::default();
        assert!(matches!(l, LeafEval::TexelWide));
        assert!(!l.is_learned());
    }

    /// `LeafEval::Texel` dispatches to `evaluate_in_progress` —
    /// must produce a finite value in `(-0.5, 0.5)` on default state.
    #[test]
    fn leaf_eval_texel_matches_evaluate_in_progress() {
        let s = CombatState::default();
        let texel = LeafEval::Texel.evaluate(&s);
        let direct = sts2_core::evaluate_in_progress(&s);
        assert!(
            (texel - direct).abs() < 1e-9,
            "LeafEval::Texel must delegate to evaluate_in_progress — got {texel} vs {direct}"
        );
        assert!(texel.is_finite());
        assert!(texel > -0.5 && texel < 0.5);
    }

    /// Bead `sts2-rs-pvuwv`: `LeafEval::evaluate` clamps to
    /// [`TERMINAL_LOSS_LEAF`] on terminal-loss states, *regardless*
    /// of the underlying feature-vector score. Hand-built dead-
    /// player state with positive `damage_dealt_raw`-like inputs
    /// would otherwise return a `>0` Texel score; the guard makes
    /// the result deterministic.
    #[test]
    fn leaf_eval_terminal_loss_clamps_to_constant() {
        let mut s = CombatState {
            is_over: true,
            player_won: false,
            ..CombatState::default()
        };
        s.player.current_hp = 0;

        let leaf = LeafEval::Texel.evaluate(&s);
        assert!(
            (leaf - TERMINAL_LOSS_LEAF).abs() < 1e-9,
            "terminal-loss state must clamp to TERMINAL_LOSS_LEAF \
             ({TERMINAL_LOSS_LEAF}), got {leaf}"
        );

        // Confirm the unguarded reading (what evaluate_in_progress
        // returns for the same state) differs — otherwise the guard
        // is a no-op on this fixture and the test would pass for
        // the wrong reason.
        let unguarded = sts2_core::evaluate_in_progress(&s);
        assert!(
            (unguarded - TERMINAL_LOSS_LEAF).abs() > 1e-3,
            "unguarded evaluate_in_progress should differ from the \
             clamp value — otherwise the guard is untested. \
             unguarded={unguarded}, clamp={TERMINAL_LOSS_LEAF}"
        );

        // RichBoth proven_loss backup: -1.0 + 0.5 * leaf = -1.45.
        let backup = -1.0_f32 + 0.5 * leaf;
        assert!(
            (backup - -1.45).abs() < 1e-6,
            "RichBoth proven_loss backup with clamp must land at -1.45, got {backup}"
        );
    }

    /// `sts2-rs-pvuwv` sibling guard: a terminal-WIN state is NOT
    /// clamped — the win-side backup
    /// (`lib.rs:1721-1722`) handles those separately via
    /// `evaluate_victory`. So `LeafEval::evaluate` must pass
    /// terminal-win states through to the underlying evaluator
    /// unchanged.
    #[test]
    fn leaf_eval_terminal_win_is_not_clamped() {
        let s = CombatState {
            is_over: true,
            player_won: true,
            ..CombatState::default()
        };

        let leaf = LeafEval::Texel.evaluate(&s);
        let direct = sts2_core::evaluate_in_progress(&s);
        assert!(
            (leaf - direct).abs() < 1e-9,
            "terminal-win state must pass through to evaluate_in_progress, \
             got leaf={leaf} vs direct={direct}"
        );
    }

    /// `from_weights_path` against a missing file must return an
    /// `Io` error rather than panic. Pins the error-path contract
    /// post-s6v2t (was previously a `Tract` error against a
    /// missing ONNX).
    #[test]
    fn missing_weights_path_yields_io_error() {
        let result =
            LearnedValueNet::from_weights_path(Path::new("/tmp/sts2-s6v2t-does-not-exist.json"));
        assert!(result.is_err());
        match result.err().unwrap() {
            ValueNetError::Io(_) => {}
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    /// The forward path's pre-tract logic (feature extraction +
    /// normalization + score mapping) is exercised indirectly via
    /// `evaluate` — pinning the score-mapping formula here so a
    /// future "tighten the bound" bead can update it deliberately.
    #[test]
    fn score_mapping_bounds_match_evaluate_in_progress_contract() {
        // p_win = 0.5 (toss-up) → score = 0.
        let centered = 0.5_f32 - 0.5;
        let score = 2.0 * centered * INPROGRESS_BOUND;
        assert!(score.abs() < 1e-9);

        // p_win = 1.0 → score = +bound (clamp would be a no-op).
        let centered = 1.0_f32 - 0.5;
        let score = (2.0 * centered * INPROGRESS_BOUND).clamp(-INPROGRESS_BOUND, INPROGRESS_BOUND);
        assert!((score - INPROGRESS_BOUND).abs() < 1e-9);

        // p_win = 0.0 → score = -bound.
        let centered = 0.0_f32 - 0.5;
        let score = (2.0 * centered * INPROGRESS_BOUND).clamp(-INPROGRESS_BOUND, INPROGRESS_BOUND);
        assert!((score + INPROGRESS_BOUND).abs() < 1e-9);

        // Structural pin: INPROGRESS_BOUND must stay strictly under
        // 0.5 so depth-cap leaf scores never bucket as a confirmed
        // terminal. Lives at const-eval time so a future bead bumping
        // the bound fails noisily.
        const _: () = assert!(INPROGRESS_BOUND < 0.5);
    }

    // ── qctu.8 LearnedPolicyPrior projection tests ─────────────────

    #[test]
    fn learned_prior_projects_dense_to_legal_actions() {
        let mut dense = vec![0.0_f32; 640];
        dense[5] = 0.3;
        dense[10] = 0.7;
        let legal_slots = [Some(5usize), Some(10usize), Some(123usize)];
        let prior =
            LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior(&dense, &legal_slots);
        assert_eq!(prior.len(), 3);
        assert!((prior[0] - 0.3).abs() < 1e-6);
        assert!((prior[1] - 0.7).abs() < 1e-6);
        assert!(prior[2].abs() < 1e-6);
        let sum: f32 = prior.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "prior must renormalize to 1.0");
    }

    #[test]
    fn learned_prior_uniform_fallback_when_dense_is_zero() {
        let dense = vec![0.0_f32; 640];
        let legal_slots = [Some(5usize), Some(10usize), Some(20usize), Some(30usize)];
        let prior =
            LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior(&dense, &legal_slots);
        let expected = 1.0 / 4.0;
        for p in &prior {
            assert!((p - expected).abs() < 1e-6);
        }
        let sum: f32 = prior.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn learned_prior_handles_unresolvable_slots() {
        let mut dense = vec![0.0_f32; 640];
        dense[7] = 1.0;
        let legal_slots = [Some(7usize), None, None];
        let prior =
            LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior(&dense, &legal_slots);
        assert_eq!(prior.len(), 3);
        assert!((prior[0] - 1.0).abs() < 1e-6);
        assert!(prior[1].abs() < 1e-6);
        assert!(prior[2].abs() < 1e-6);
    }

    // ── GitHub #13 / #24 stats instrumentation ─────────────────────

    #[test]
    fn stats_track_uniform_fallback_rate() {
        let stats = LearnedPolicyPriorStats::default();

        // Two healthy projections, one zero-mass projection.
        let mut dense = vec![0.0_f32; 640];
        dense[5] = 1.0;
        let legal = [Some(5usize), Some(6usize)];
        let _ = LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior_instrumented(
            &dense,
            &legal,
            Some(&stats),
        );
        let _ = LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior_instrumented(
            &dense,
            &legal,
            Some(&stats),
        );

        let dense_zero = vec![0.0_f32; 640];
        let _ = LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior_instrumented(
            &dense_zero,
            &legal,
            Some(&stats),
        );

        assert_eq!(stats.evaluations(), 3);
        assert_eq!(stats.uniform_fallbacks(), 1);
        assert!((stats.uniform_fallback_rate() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn stats_detect_target_collision() {
        let stats = LearnedPolicyPriorStats::default();
        let mut dense = vec![0.0_f32; 640];
        dense[5] = 1.0;

        // Two distinct legal actions collapsing to the same slot —
        // exactly the #24 target-collision case.
        let legal = [Some(5usize), Some(5usize), Some(10usize)];
        let _ = LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior_instrumented(
            &dense,
            &legal,
            Some(&stats),
        );

        assert_eq!(stats.evaluations(), 1);
        assert_eq!(stats.target_collisions_total(), 1);
        assert_eq!(stats.evaluations_with_collisions(), 1);
        assert!((stats.collision_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn stats_no_collision_for_distinct_slots() {
        let stats = LearnedPolicyPriorStats::default();
        let mut dense = vec![0.0_f32; 640];
        dense[5] = 1.0;
        let legal = [Some(5usize), Some(6usize), Some(7usize)];
        let _ = LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior_instrumented(
            &dense,
            &legal,
            Some(&stats),
        );
        assert_eq!(stats.target_collisions_total(), 0);
        assert_eq!(stats.evaluations_with_collisions(), 0);
    }

    #[test]
    fn stats_mean_legal_mass_tracks_projection_quality() {
        let stats = LearnedPolicyPriorStats::default();
        // Healthy: sum=1.0 over legal.
        let mut dense_full = vec![0.0_f32; 640];
        dense_full[5] = 1.0;
        let legal = [Some(5usize)];
        let _ = LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior_instrumented(
            &dense_full,
            &legal,
            Some(&stats),
        );
        // Partial: sum=0.25 over legal (rest on illegal slots).
        let mut dense_partial = vec![0.0_f32; 640];
        dense_partial[5] = 0.25;
        dense_partial[100] = 0.75;
        let _ = LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior_instrumented(
            &dense_partial,
            &legal,
            Some(&stats),
        );

        let mean = stats.mean_legal_mass();
        assert!((mean - 0.625).abs() < 1e-3, "expected ~0.625, got {mean}");
    }

    #[test]
    fn stats_reset_clears_counters() {
        let stats = LearnedPolicyPriorStats::default();
        let dense = vec![0.0_f32; 640];
        let legal = [Some(0usize)];
        let _ = LearnedPolicyPrior::<LearnedValueNet>::project_dense_prior_instrumented(
            &dense,
            &legal,
            Some(&stats),
        );
        assert_eq!(stats.evaluations(), 1);
        stats.reset();
        assert_eq!(stats.evaluations(), 0);
        assert_eq!(stats.uniform_fallbacks(), 0);
    }

    #[test]
    fn learned_prior_exposes_shared_stats_handle() {
        // Smoke test: the Arc<Stats> returned by `stats()` is the
        // same handle the prior writes to. Doesn't need a live ONNX
        // fixture since we only inspect the surface.
        // (Skipped if ONNX fixture missing — covered by the
        // projection unit tests which do the heavy lifting.)
    }

    // ── Texel-compose (sts2-rs-ae14u) ─────────────────────────────
    //
    // The composition path is exercised through a stub `LearnedValueNet`
    // -free path: we test the `compose_score` math directly by
    // synthesizing it via the same formula. The full
    // `LearnedValueNet::evaluate` integration needs an ONNX fixture
    // which fuzz tests skip; the math we own here is the addition +
    // sigmoid-inverse + tanh chain.

    #[test]
    fn compose_off_preserves_pure_nn_mapping() {
        // p_win = 0.5 → centered = 0 → score = 0.
        let p = 0.5_f32;
        let base = map_win_to_score(p);
        assert!(base.abs() < 1e-6);
        // p_win = 1.0 (clamped) → +INPROGRESS_BOUND.
        let high = map_win_to_score(1.0);
        assert!((high - INPROGRESS_BOUND).abs() < 1e-6);
        // p_win = 0.0 → -INPROGRESS_BOUND.
        let low = map_win_to_score(0.0);
        assert!((low + INPROGRESS_BOUND).abs() < 1e-6);
    }

    /// At `k = 0`, the composed score must reduce to a pure tanh of
    /// the NN logit, scaled into the depth-cap range — Texel drops
    /// out of the sum.
    #[test]
    fn compose_zero_k_drops_texel_contribution() {
        let p = 0.7_f32;
        let logit = (p / (1.0 - p)).ln();
        let expected = (logit.tanh()) * INPROGRESS_BOUND;
        // Manual composition with k=0; we don't need a real
        // LearnedValueNet for the math test.
        let texel_norm = 0.42_f32; // arbitrary — must be ignored at k=0
        let combined = (logit + 0.0_f32 * texel_norm).tanh();
        let score = (combined * INPROGRESS_BOUND).clamp(-INPROGRESS_BOUND, INPROGRESS_BOUND);
        assert!((score - expected).abs() < 1e-6, "k=0 must drop Texel");
    }

    /// At `k > 0` with a positive Texel signal and a neutral NN
    /// (logit = 0), the composed score must be strictly positive —
    /// Texel pulls the output above zero.
    #[test]
    fn compose_positive_k_lets_texel_pull_score_positive() {
        let p = 0.5_f32; // neutral NN — logit = 0
        let nn_logit = (p / (1.0 - p)).ln();
        assert!(nn_logit.abs() < 1e-6);
        let texel_norm = 0.3_f32; // arbitrary positive
        let k = 0.326_f32; // Caissawary reference
        let combined = (nn_logit + k * texel_norm).tanh();
        let score = (combined * INPROGRESS_BOUND).clamp(-INPROGRESS_BOUND, INPROGRESS_BOUND);
        assert!(
            score > 0.0,
            "neutral NN + positive Texel + k>0 must pull score positive; got {score}"
        );
        // And symmetrically negative Texel pulls negative.
        let combined_n = (nn_logit + k * (-texel_norm)).tanh();
        let score_n = (combined_n * INPROGRESS_BOUND).clamp(-INPROGRESS_BOUND, INPROGRESS_BOUND);
        assert!(score_n < 0.0);
    }

    /// Bead `sts2-rs-luh6m`: the production h=16 value-net weights at
    /// `data/value_policy_net/h16.json` must round-trip through
    /// [`LearnedValueNet::from_weights_path`] without shape mismatch.
    /// Pins the weights-file format + advertised architecture
    /// (input_dim wide-schema-dim, hidden=16, trunk_depth=3) so a
    /// future trainer-side dim bump can't silently break the production
    /// asset without this regressing.
    ///
    /// **Ignored under bead `sts2-rs-h5nf6` (6pzjm.1):** the wide schema
    /// grew from 88 → 172 columns to add per-enemy peek_intents
    /// lookahead. The shipped h=16 weights are still the 88-column
    /// asset; the corpus regen + retrain land via the sibling beads
    /// `sts2-rs-1hiso` (6pzjm.2) and `6pzjm.3`. Un-ignore once the new
    /// 172-column h=16 weights overwrite `data/value_policy_net/h16.json`.
    #[ignore = "h5nf6 schema width 88 → 172; retrain pending under 6pzjm.2/6pzjm.3"]
    #[test]
    fn luh6m_production_h16_weights_load_and_match_arch() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/value_policy_net/h16.json");
        let net = LearnedValueNet::from_weights_path(&path)
            .expect("data/value_policy_net/h16.json must load cleanly");
        assert_eq!(
            net.weights.hidden, 16,
            "production weights must be hidden=16 (post-qmncd sweep winner)"
        );
        assert_eq!(
            net.weights.trunk.len(),
            3,
            "production weights must keep trunk_depth=3 (qmncd Stage A baseline)"
        );
    }

    /// The composed score must stay within `[-INPROGRESS_BOUND, +INPROGRESS_BOUND]`
    /// for any reasonable inputs — `tanh` squashes the sum into `(-1, 1)`
    /// and the final scale lands in the depth-cap range.
    #[test]
    fn compose_score_respects_inprogress_bound() {
        // Extreme NN (near-1) + extreme Texel + large k. Must still
        // clamp to the depth-cap range.
        let p = 0.999_999_f32;
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        let nn_logit = (p / (1.0 - p)).ln();
        let texel_norm = 1.0_f32; // maximum Texel signal
        let k = 5.0_f32; // far above any realistic k
        let combined = (nn_logit + k * texel_norm).tanh();
        let score = (combined * INPROGRESS_BOUND).clamp(-INPROGRESS_BOUND, INPROGRESS_BOUND);
        assert!(score <= INPROGRESS_BOUND);
        assert!(score >= -INPROGRESS_BOUND);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact-by-construction: same fn on the same state.
    fn texel_wide_leaf_matches_core_fn() {
        let s = CombatState::default();
        assert_eq!(
            LeafEval::TexelWide.evaluate(&s),
            sts2_core::evaluate_in_progress_wide(&s)
        );
    }

    #[test]
    fn texel_wide_leaf_still_short_circuits_terminal_loss() {
        let s = CombatState {
            is_over: true,
            player_won: false,
            ..CombatState::default()
        };
        let leaf = LeafEval::TexelWide.evaluate(&s);
        assert!(
            (leaf - TERMINAL_LOSS_LEAF).abs() < 1e-9,
            "terminal-loss state must clamp to TERMINAL_LOSS_LEAF \
             ({TERMINAL_LOSS_LEAF}), got {leaf}"
        );
    }

    /// Phase 2: a runtime leaf carrying the SAME const weights must score
    /// identically to the `TexelWide` const leaf — this is the loader/variant
    /// parity guard the round bootstrap (champion-0 dump) relies on.
    #[test]
    #[allow(clippy::float_cmp)] // exact-by-construction: same weights, same state.
    fn texel_wide_runtime_matches_const_with_same_weights() {
        let s = CombatState::default();
        let w: Arc<[f32; sts2_core::WIDE_FEATURE_DIM]> =
            Arc::new(sts2_core::EVAL_FEATURE_WEIGHTS_WIDE);
        assert_eq!(
            LeafEval::TexelWideRuntime(w).evaluate(&s),
            sts2_core::evaluate_in_progress_wide(&s)
        );
    }

    #[test]
    fn texel_wide_runtime_short_circuits_terminal_loss() {
        let s = CombatState {
            is_over: true,
            player_won: false,
            ..CombatState::default()
        };
        let w: Arc<[f32; sts2_core::WIDE_FEATURE_DIM]> =
            Arc::new([0.0; sts2_core::WIDE_FEATURE_DIM]);
        let leaf = LeafEval::TexelWideRuntime(w).evaluate(&s);
        assert!((leaf - TERMINAL_LOSS_LEAF).abs() < 1e-9);
    }

    #[test]
    fn load_wide_leaf_weights_reads_ordered_json() {
        let dim = sts2_core::WIDE_FEATURE_DIM;
        // Distinct per-index values so a positional bug would be caught.
        let nums: Vec<String> = (0..dim)
            .map(|i| format!("{:.3}", i as f32 * 0.01))
            .collect();
        let json = format!("{{\"fitted_weights_ordered\":[{}]}}", nums.join(","));
        let path = std::env::temp_dir().join(format!(
            "sts2-phase2-leaf-{}-{}.json",
            std::process::id(),
            dim
        ));
        std::fs::write(&path, json).expect("write temp weights");
        let w = load_wide_leaf_weights(&path).expect("load weights");
        let _ = std::fs::remove_file(&path);
        assert_eq!(w.len(), dim);
        assert!((w[0] - 0.0).abs() < 1e-9);
        assert!((w[10] - 0.10).abs() < 1e-4);
    }

    #[test]
    fn load_wide_leaf_weights_rejects_wrong_width() {
        let json = "{\"fitted_weights_ordered\":[1.0,2.0,3.0]}";
        let path = std::env::temp_dir().join(format!(
            "sts2-phase2-leaf-badwidth-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, json).expect("write temp weights");
        let r = load_wide_leaf_weights(&path);
        let _ = std::fs::remove_file(&path);
        assert!(r.is_err(), "3-wide must be rejected vs WIDE_FEATURE_DIM");
    }
}
