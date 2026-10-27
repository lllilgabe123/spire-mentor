# Ironclad a10_it700 standard strength eval on the GH #830 default (pre-registered 2026-10-26T01:41+01:00)

Purpose: the playbook §2.6 re-measure the ship PR (#942) asked for. Not a gate.

## Build
- main `a1bd07c7193ce2f0ee303250d57573b52541b986` (#942 merge `118085173` + #938 docs merge).
- Worktree `.worktrees/ic-bundle-default`, detached at that commit.
- Wheel: PGO venv `.venv-h/4e5bb730d28e2976`. Its `.so` sha256 is `823fa393eab74db57d0cffd6f71c4a5c873a74d345a354adfca8458bf1333ba2` (= PGO_SO_SHA256 of the ship identity archive). `build_git` is `687e16ccd`, a docs-only ancestor with the same engine source.
- Built-in prior: `builtin_prior_net_json("Ironclad")` sha `ca469259…d4f2`.
- Checkpoint: `sts2-pycli/data/ppo/ppo_fullrun_ironclad_a10_it700.pt`, sha `91c83c1a…44fc`.

## Arms (run in sequence, D first; 30 procs each; every other STS2_* unset)
- **D (deliverable):** the §2.6 command verbatim: `STS2_RUNAI_MCTS_OVERRIDES="removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0"`, `--policy net_greedy --character Ironclad --ascension 10 --act1 roll --seeds 400 --seed-start 91000000 --procs 30 --sims 4000 --boss-sims 16000 --run-timeout 14400`. This is the shipped Ironclad default; the env line is redundant (identity arm E ≡ D).
- **P (pre-ship setting on this wheel):** D plus `STS2_RUNAI_PRIOR_NET=off`. This is the knobs only, the setting of the old 23.0% row, with rule 5c now included.

## Analysis (fixed now)
1. Filter error and timeout rows before anything else. Report the counts. Pair only seeds valid in both arms.
2. The playbook row is D's A10 win rate and its A9 estimate (`reached_second_boss`), via `docs/experiments/ironclad-a10/a10_score.py`, with Wilson 95% CIs.
3. D vs P, descriptive: a paired difference with discordant counts b/c, an exact McNemar p and a 95% CI, for both A10 and the A9 estimate. This is n=400 at 4× sims, so it is not a gate and promotes nothing. Whatever comes out, the ship decision stands (both A10 gates at 1000/4000).
4. Cost: wall time per arm and CPU per run, D/P.
5. The old row (23.0% / 49.8%, older wheel, other host) is quoted, not paired: its per-seed rows are not on this host.

Raw data: `runs/`. Logs: `logs/`. Driver: `scripts/run_eval.sh`.
