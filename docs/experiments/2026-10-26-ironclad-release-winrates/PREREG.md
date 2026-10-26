# Ironclad release win rates on the GH #830 default (pre-registered 2026-10-26)

Purpose: re-measure the Ironclad column of `docs/experiments/2026-10-11-release-winrates.md` (the numbers on `docs/site/index.html` and in `release/public-overlay/README.md`) for the current Ironclad champion under the shipped Ironclad search default. Owner asked for this on 2026-10-26. It is a release measurement, not a gate.

## Protocol: identical to the 2026-10-11 sweep except for champion and engine
- Driver: `ppo.probe --policy net_greedy --act1 roll --character Ironclad`. Combat search is 8000 sims per decision, boss fights included (`--sims 8000 --boss-sims 8000`), in one tree. Run timeout 14400 s.
- Seeds: `--seed-start 81000000` at every level. 200 seeds at A0–A8, 500 at A9–A10. This is the same panel as 2026-10-11, so each run pairs by seed with the old Ironclad cell.
- Order: A10, A9, then A0, A6, A3, A8, A1, A5, A2, A4, A7. One cell at a time, 30 procs (2026-10-11 used 32; the proc count does not change results).
- Champion: `ppo_fullrun_ironclad_a10_it700.pt` (sha 91c83c1a…), from `champions.toml` at main.
- Engine: main `0181fa552` (#942 + #938 + #943 merged). PGO venv `.venv-h/4e5bb730d28e2976`, `.so` sha `823fa393…`, built-in Ironclad prior sha `ca469259…`. No `STS2_*` variables are set, so the run uses the shipped Ironclad default: knobs `removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0` in every fight, plus the BC move prior (mix 0.5, T 1) in elite and boss fights.

## Analysis (fixed now)
1. Filter error and timeout rows first and report their counts. A timeout is not a loss.
2. Per level, report the win rate with a Wilson 95% interval, plus the unweighted mean over A0–A10 and the act-1 split (pooled, by seed). These use the same definitions as 2026-10-11, so the numbers are comparable to its tables.
3. Compare with the 2026-10-11 Ironclad column, paired by seed, as a descriptive read only. For each cell, give the discordant counts b/c, the exact McNemar p, and the diff with a paired 95% CI. The pooled A0–A10 result gets a seed-clustered caveat: the same seeds repeat across levels. The comparison CONFOUNDS the champion (plr_it550 → a10_it700), the search default (#830), and every engine change between 0cab615a3 and 0181fa552. It attributes nothing.
4. Cost: the wall time per run for each cell.

Outputs: `cells/ironclad_A<asc>.{jsonl,log}`, `progress.log`, `analysis/`.
