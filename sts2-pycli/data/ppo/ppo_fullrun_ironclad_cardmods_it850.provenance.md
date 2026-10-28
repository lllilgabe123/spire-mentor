# PPO champion — `ppo_fullrun_ironclad_cardmods_it850`

**Passed its pre-registered A10 gate 2026-10-27; promoted by user decision**
on the A10 objective set 2026-10-23.

- **Character**: Ironclad
- **Source**: `~/ppo-ironclad-cardmods/main.iter_850.pt`, copied byte for byte
  (the exact file every gate arm loaded). Checkpoints are named by `it+1`: the
  net after the final logged iteration `it849` is `iter_850`.
- **Replaces**: `ppo_fullrun_ironclad_a10_it700` (promoted 2026-10-23 by user
  decision on the A10 objective).
- **Features**: `card_mods,reward_upgrade`, from the checkpoint's own `cfg`.
  First Ironclad champion trained with a non-default schema. `card_mods`
  (GH #947) puts each card's enchantment id in `cat[11]` and its amount in
  cont slot 25 (N_CONT 26); `reward_upgrade` (GH #545/#726) shows card-reward
  options' true upgraded flag in `cat[1]`. Serving this net at the default
  schema would hand it inputs it was not trained on and nothing would error:
  the Rust lane reads the spec from `champions.toml` (`features_for`), Python
  from the checkpoint cfg.
- **Net**: `ppo.model.RunPolicyNet` defaults (dim=192, n_layers=3, n_heads=4);
  `cont_proj` input 26 slots × 12 sinusoid columns.
- **Pre-registration and record**:
  `docs/experiments/2026-10-27-ironclad-a10-cardmods.md`; harness in
  `docs/experiments/ironclad-a10-cardmods/`.

## The recipe, and what changed

A **true resume** (`--resume-from-iter 700`, optimizer state kept) of the
parent, it700 → it850. Declared changes versus the parent's stored cfg:

- `--features card_mods,reward_upgrade` (with `STS2_PPO_FEATURES` exported).
  `card_mods` appends a cont slot, so
  `docs/experiments/ironclad-a10-cardmods/widen_resume_ckpt.py` widened
  `cont_proj.proj.weight` [192, 300] → [192, 312] with zero columns and zero
  AdamW moments; the widened net computed exactly what the parent computed
  (asserted).
- `--iterations 850`: the horizon, which re-opened the absolute-iteration LR
  anneal (`lr_frac` 0.423 at it700 → 0.30 at it849).

Everything else matched the parent: A10 only (`--ascension-weights "10:1"`),
PLR off, 960 games/iter, 250/1000 training sims, entropy at 0.0125, 2 epochs,
`--act1 roll`, trained with `STS2_RUNAI_POTION_SWAP=1` and **served with it
OFF**.

**The combat search changed under the parent.** `a10_it700` was trained before
the GH #830 Ironclad search shipped (forms + combo knobs; BC move prior in
elite and boss fights). This leg trained under it.

**Attribution: none.** The two encoder features, 150 more A10 iterations, the
LR re-open and the #830 search are confounded. There was no feature-off arm.

## Training trajectory

2026-10-27 00:39 → 15:22, ~350 s/iter, 0 errored games in 144,000.
`dec_per_run/card-enchant` 0.45. At it780 `emb_ench` showed row-specific drift
(median 0.023, max 0.052) and the new `cont_proj` columns had left zero (mean
|w| 0.0018 vs 0.039 for the old ones), so both features reached the net.

Greedy probes (512 seeds, 250/1000 sims, trend only): A10 it700 13.5% → peak
it800 19.3% → it850 15.0%; A0 held at 89–94%. The candidate was fixed in
advance as the final checkpoint, not picked from the probes.

## The gate

Both arms: one binary (wheel src_hash `aa7acaa26a2bf981`, main `b1fb7ef9b`),
`net_greedy`, deployed eval budget 1000/4000 with the shipped #830 Ironclad
search, `--act1 roll`, potion-swap OFF, torch `.pt` inference, each arm
featurized at its own checkpoint cfg. **0 errored rows in 2,800.**

| gate | champion a10_it700 | it850 | Δ | discordant (cand/champ) | McNemar p |
|---|---|---|---|---|---|
| **A10 × 1000**, panel 133,000,000 (primary) | 21.7% | 28.1% | **+6.40pp** | 163/99 | **9.2e-05** |
| A9 estimate from those A10 runs | 48.7% | 55.4% | +6.70pp | 200/133 | 2.9e-04 |
| true A9 × 400, panel 133,100,000 (readout) | 49.5% | 51.2% | +1.75pp | 68/61 | 0.60 |

Rule: PASS iff Δ > 0 and p < 0.01. **PASS.**

## Standard strength eval (playbook §2.6)

4000/16000 sims (the advisor's default budget), A10 × 400, seeds 91,000,000+,
wheel `aa7acaa26a2bf981`: **30.00% A10** [25.72, 34.66], **56.50% A9
estimate** [51.60, 61.27]. The outgoing net on the same seeds (wheel
`4e5bb730d28e2976`): 26.75% / 52.25%. Paired +3.25pp (66/53, p 0.27);
different wheels, so descriptive.

## Serving checks

- **ONNX** exported from this checkpoint by `ppo.export_onnx`
  (`external_data=False`, no `.onnx.data` sidecar). The gate swept the torch
  `.pt`; the Rust gate B checks the ONNX against torch π/V on fixture rows
  dumped from this checkpoint at `card_mods,reward_upgrade`
  (`ppo_encoder_parity_card_mods_reward_upgrade.jsonl`).
- **Overflow headroom**: max deck 53 across 1,800 candidate runs (gate +
  standard eval), against the `obs + act ≤ T_MAX` binder at deck ≈ 126.
- **Known gap**: the advisor wire does not carry a card's pending enchantment
  in every screen (GH #574); there the net reads the card as unenchanted.
