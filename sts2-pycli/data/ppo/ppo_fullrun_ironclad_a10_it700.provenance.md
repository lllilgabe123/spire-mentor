# PPO champion — `ppo_fullrun_ironclad_a10_it700`

**Promoted 2026-10-23 by user decision over a FAILED pooled gate**, on the A10
objective: "our main goal is improvement at A10 because that's what pro
players want." The A10 and A9 gains are large and replicate on three
independent panels. The pre-registered pooled A0–A10 rule failed on
significance (+1.59pp, p = 0.107) while passing its regression guard (worst
level −3.0pp against −5.0pp).

- **Character**: Ironclad
- **Source**: `~/ppo-ironclad-a10/main.iter_700.pt`, copied byte for byte (the
  exact file every gate arm loaded). Checkpoints are named by `it+1`: the net
  after the final logged iteration `it699` is `iter_700`.
- **Replaces**: `ppo_fullrun_ironclad_plr_it550_v0111vocab` (promoted
  2026-09-02 on a passed gate; vocab re-export 2026-10-18, GH #799)
- **Features**: `""` (default schema), from the checkpoint's own `cfg`.
- **Net**: `ppo.model.RunPolicyNet` defaults (dim=192, n_layers=3, n_heads=4);
  card/relic/potion tables 601/301/66 rows (the v0.111.0 vocab).
- **Pre-registration and record**:
  `docs/experiments/2026-10-22-ironclad-a10-continuation.md`; harness in
  `docs/experiments/ironclad-a10/`.

## The recipe, and what changed

A **true resume** (`--resume-from-iter 550`, optimizer state kept) of the
parent, it550 → it700. Declared changes versus the parent's stored cfg:

- `--ascension-weights "10:1"`: every training game at A10.
- `--plr-replay-p 0`: Prioritized Level Replay off (it measured NULL on Silent).
- `--iterations 700`: the horizon, which re-opened the absolute-iteration LR
  anneal (`lr_frac` 0.449 at it550 → 0.30 at it699).

Everything else matched the parent: 960 games/iter, 250/1000 training sims,
entropy at its final 0.0125, 2 epochs, `--act1 roll`, trained with
`STS2_RUNAI_POTION_SWAP=1` and **served with it OFF**.

The parent's `_v0111vocab.pt` is a byte copy of the old-vocab checkpoint, so a
strict resume refused it. `docs/experiments/ironclad-a10/widen_resume_ckpt.py`
widened the weights (`load_compat`) and re-laid the AdamW moments of the three
grown tables by the same rule, then verified a strict load and bit-identical
rows for every id the parent knew.

**The combat engine changed under the parent.** `plr_it550` was trained with
heuristic playouts. This leg trained under the #816 Ironclad `rl2` playout net
(the BeliefDraw search's default since 2026-10-18) and the #771 last-boss
objective. Part of any gain may be the run policy adapting to a stronger
combat pilot.

**Attribution: none.** A10-only weighting, PLR off, the horizon/LR re-anneal
and the engine change are perfectly confounded. There was no control arm.

## Training trajectory

Greedy probes (512 seeds, 250/1000 sims; A9 read off the A10 runs) sat within
noise of the it550 baseline (A10 10.2%) through it670, then rose over the last
30 iterations as the LR annealed: it680 15.6%, it690 13.9%, it700 15.8% (A9
estimate 35.4%, A0 88.7% vs 85.5%). Entropy fell 0.41 → 0.30; KL stayed
≈ 0.005 and clip fraction ≈ 0.04. 0 errored games in 144,000. The candidate
was fixed in advance as the final checkpoint, not picked from the probes.

## The gates

All arms: one binary (wheel src_hash `457d7a9d3b9823a2`, origin/main
`894213c42`), `net_greedy`, deployed eval budget 1000/4000, `--act1 roll`,
potion-swap OFF, torch `.pt` inference. **0 errored rows in 7,200.**

> **2026-10-25: the deployed Ironclad search changed after these gates (GH #830,
> commit `a70aa8bec`).** The gates above ran the pre-ship search: the BeliefDraw
> dialect, no forms or combo knobs, no move prior. The run agent now plays
> Ironclad with the forms + combo knobs in every fight and a built-in BC move
> prior in elite and boss fights. That bundle was gated on this champion: A10,
> 1000/4000, 1,500 paired runs, +5.20 ± 1.18pp (19.5% → 24.7%), repeated
> 97.06% CI [+2.62, +7.78]. A second, independent A10 gate of the same bundle
> on 4,000 paired runs measured +3.45pp (19.5% → 22.9%, McNemar p 8.8e-07).
> To reproduce the numbers in this file on a newer wheel, set
> `STS2_RUNAI_MCTS_OVERRIDES=removed_counts_forms=0,endturn_widening_c=5.787,draw_widening_c=2.0`
> and `STS2_RUNAI_PRIOR_NET=off`.

| gate | champion | it700 | Δ | discordant | McNemar p |
|---|---|---|---|---|---|
| **A10 × 1000**, panel 131,000,000 (primary) | 12.2% | 19.1% | **+6.90pp** | 132/63 | 8.7e-07 |
| A9 estimate from those A10 runs | 39.4% | 45.0% | +5.60pp | 191/135 | 0.0023 |
| **true A9 × 400**, panel 131,100,000 | 33.8% | 44.8% | **+11.0pp** | 88/44 | 1.6e-04 |
| pooled A0–A10 × 200, panel 131,200,000 | 74.68% | 76.27% | +1.59pp | 240/205 | 0.107 |

Discordant is candidate-only / champion-only. Pooled panel per level:

| lvl | champion | candidate | Δ | b/c | p |
|---|---|---|---|---|---|
| A0 | 92.0% | 92.0% | +0.00pp | 11/11 | 1.00 |
| A1 | 94.5% | 91.5% | −3.00pp | 8/14 | 0.29 |
| A2 | 91.0% | 92.5% | +1.50pp | 12/9 | 0.66 |
| A3 | 88.5% | 90.0% | +1.50pp | 18/15 | 0.73 |
| A4 | 88.5% | 90.5% | +2.00pp | 18/14 | 0.60 |
| A5 | 82.0% | 79.5% | −2.50pp | 21/26 | 0.56 |
| A6 | 84.0% | 88.5% | +4.50pp | 30/21 | 0.26 |
| A7 | 79.5% | 81.5% | +2.00pp | 26/22 | 0.67 |
| A8 | 73.0% | 73.5% | +0.50pp | 32/31 | 1.00 |
| A9 | 38.0% | 41.0% | +3.00pp | 37/31 | 0.55 |
| A10 | 10.5% | 18.5% | **+8.00pp** | 27/11 | 0.014 |

Seed-clustered sign-flip p = 0.20 (design effect 1.58×). A9 estimate from the
pooled A10 cells: 33.0% → 41.5%, +8.5pp.

**Reading.** A clear A10/A9 improvement (+6.9 to +8.0pp at A10, +3 to +11pp at
A9, every panel positive) and no detectable change at A0–A8. Deaths moved from
act 1 (100 → 70 across the pooled panel) into act 2 (133 → 178), with fewer
act-3 deaths (324 → 274).

## Serving checks

- **ONNX** exported from this checkpoint by `ppo.export_onnx`
  (`external_data=False`, no `.onnx.data` sidecar). The gates swept the torch
  `.pt`, so "ship the exact `.onnx` the gate swept" cannot hold literally; the
  Rust parity gates (fixtures re-dumped at this checkpoint) check the ONNX
  against torch instead. Graph I/O and all 87 initializer shapes match the
  outgoing champion's graph.
- **Overflow headroom**: max deck 50 across 3,587 candidate gate runs, against
  the `obs + act ≤ T_MAX` binder at deck ≈ 126. The torch path records no
  `ppo_peak_*` fields, so deck size is the proxy.
- **Engine drift**: gated on `894213c42`; promoted onto a main that includes
  #922 (Ironclad Thrash after Gigantification) and the #830 turn-plan knob
  (default OFF). Neither changes the run-policy encoder.
