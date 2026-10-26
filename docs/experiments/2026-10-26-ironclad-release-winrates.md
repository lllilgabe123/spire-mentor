# Ironclad release win rates on `a10_it700` and the GH #830 default (2026-10-26)

**Status: done (2026-10-26).** 2,800 runs, 0 errored, 0 timed out. This
re-measures the Ironclad column of the release win rates
(`2026-10-11-release-winrates.md`) with the current champion and the shipped
Ironclad search. It is a release measurement, not a gate: it changes no
decision about the champion or the search.

`PR #…` and `GH #…` numbers and commit hashes refer to the private development
repository. Where this write-up relies on a private record, it states the
fact here. The sibling folder `2026-10-26-ironclad-release-winrates/` holds
the per-seed rows, the 2026-10-11 Ironclad rows they pair with, the driver
and the analysis scripts, and **Reproduce** below reruns the sweep from a
clone of the public repository.

## Purpose

The release win rates on the docs site, in the public README and in
`2026-10-11-release-winrates.md` come from the 2026-10-11 sweep (run 11–13
October, engine `0cab615a3`). Since then three things changed for Ironclad.
The Ironclad `rl2` playout net became the default playout of its combat
search (PR #816, 2026-10-18). The champion became `a10_it700` (promoted
2026-10-23). And the run agent's Ironclad search gained the GH #830 default
(PR #942, merged 2026-10-26): the forms + combo knobs in every fight and a
built-in BC move prior in elite and boss fights. The pages said that the
Ironclad row predated the champion and the #830 search and had not been
re-measured. This sweep runs the 2026-10-11 protocol on the same seeds with
the current champion and the shipped default, so the pages can carry current
Ironclad numbers. The analysis was fixed in `PREREG.md` before the runs.

The other four characters were not re-run. They kept their champions (the
same weights; GH #799 re-exported them for the v0.111 vocabulary on
2026-10-18), but each one's combat search has shipped a built-in playout net
since 2026-10-21 (GH #822 Regent, #823 Necrobinder, #824 Silent, #825
Defect). Their 2026-10-11 columns are the latest full release sweep for
those characters, not a measurement of their current stack (Caveats).

## Setup

| | |
|---|---|
| Code | private main `0181fa552` (PR #942, #938 and #943 merged). Its engine source is the engine of the public commit that first carried `2026-10-26-ic-a10-it700-standard-eval.md`; a wheel built from the public tree reports `src_hash` `6a98001ae326637b` (that write-up, Reproduce step 1) |
| Wheel | PGO build, `src_hash` `4e5bb730d28e2976`, `.so` sha256 `823fa393…3ba2`, `py_src` `774d91045cdb5433`: the wheel of the §2.6 standard eval |
| Run policy | `a10_it700`, `net_greedy`. The run loaded the torch checkpoint `ppo_fullrun_ironclad_a10_it700.pt` (sha256 `91c83c1a…44fc`), which the public snapshot does not ship; the standard eval's `scripts/onnx_to_ckpt.py` rebuilds it from the shipped `.onnx` |
| Combat search | the shipped Ironclad default, with no `STS2_*` variable set: honest BeliefDraw with the Ironclad `rl2` playout net (compiled in); the knobs `removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0` in every fight; the built-in BC move prior (`sts2-rs/data/prior_nets/ironclad.json`, sha256 `ca469259…`, compiled in) at mix 0.5, T 1 in elite and boss fights. 8,000 sims per decision, boss fights included (`--sims 8000 --boss-sims 8000`), one search tree |
| Driver | `ppo.probe --policy net_greedy --character Ironclad --act1 roll --seed-start 81000000 --sims 8000 --boss-sims 8000 --run-timeout 14400 --procs 30` (`run.sh`) |
| Panel | the 2026-10-11 panel: seeds 81,000,000–81,000,199 at A0–A8 (200 each) and 81,000,000–81,000,499 at A9–A10 (500 each), 2,800 runs. Act 1 is rolled per seed: 1,297 Overgrowth and 1,503 Underdocks runs, the same split as on 2026-10-11 |
| Order | one level at a time: A10, A9, A0, A6, A3, A8, A1, A5, A2, A4, A7 |
| Env | every `STS2_*` variable unset; BLAS threads 1 |
| Host | a shared 48-CPU Linux machine, 30 processes, 2026-10-26 09:26–14:08 CET |

The protocol is the 2026-10-11 one with two changes: the champion and the
engine (search default included). That sweep ran 32 processes; a run depends
on its seed alone, not on the process count.

## Results

All 2,800 rows: outcome `won` or `lost`, 0 error, 0 timeout, nothing filtered.
Every level holds exactly its seed range, once.

| Asc | n | wins | win rate % [Wilson 95%] |
|---|---|---|---|
| A0 | 200 | 192 | 96.0 [92.3–98.0] |
| A1 | 200 | 198 | 99.0 [96.4–99.7] |
| A2 | 200 | 187 | 93.5 [89.2–96.2] |
| A3 | 200 | 188 | 94.0 [89.8–96.5] |
| A4 | 200 | 190 | 95.0 [91.0–97.3] |
| A5 | 200 | 183 | 91.5 [86.8–94.6] |
| A6 | 200 | 187 | 93.5 [89.2–96.2] |
| A7 | 200 | 177 | 88.5 [83.3–92.2] |
| A8 | 200 | 169 | 84.5 [78.8–88.9] |
| A9 | 500 | 282 | 56.4 [52.0–60.7] |
| A10 | 500 | 136 | 27.2 [23.5–31.3] |
| **mean A0–A10** | | | **83.6** |

Every rate: Ironclad, `a10_it700` `net_greedy`, 8,000 sims per decision in
one tree with boss fights included, the shipped GH #830 default, engine
`0181fa552`, seeds 81,000,000+. The mean is the unweighted mean of the 11
level rates (83.55), as on 2026-10-11, and the Wilson intervals use
z = 1.96, the 2026-10-11 constant.

- A0–A8 sit between 84.5% and 99.0%; A1 above A0 is inside the noise, as it
  was on 2026-10-11.
- The cliff is still at A9 (−28.1pp from A8), and A10 roughly halves A9
  again. A9 adds Deadly Enemies; A10 adds a second act-3 boss.
- Where the A10 losses ended: 36 in act 1, 79 in act 2, 249 in act 3 (7.2%,
  15.8% and 49.8% of the 500 runs). At A9: 24, 81 and 113.

### Act 1 split

| | Overgrowth (n = 1,297) | Underdocks (n = 1,503) |
|---|---|---|
| 2026-10-26: Ironclad `a10_it700`, #830 default | 957 = 73.8% [71.3–76.1] | 1,132 = 75.3% [73.1–77.4] |
| 2026-10-11: Ironclad `plr_it550` | 820 = 63.2% [60.6–65.8] | 977 = 65.0% [62.6–67.4] |

Both rows: 8,000 sims, pooled over the 11 levels by run (the 2026-10-11
definition). The act is a function of the seed, so this is a by-seed split,
not a paired comparison of acts, and the pooled split carries the level mix:
the seeds that play only A9 and A10 (81,000,200–499) are 159/300 Overgrowth,
against 89/200 below them. With the 11 levels weighted equally, the gap goes
away in both sweeps (83.5% vs 83.5% now, 74.8% vs 74.8% on 2026-10-11).

## Compared with the 2026-10-11 column

Paired by seed against the published Ironclad column (champion `plr_it550`,
the same driver, budget, seeds and n, engine `0cab615a3`). This is
descriptive: it changes the champion (`plr_it550` → `a10_it700`), the search
default (GH #830) and the engine (690 commits from `0cab615a3` to
`0181fa552`, among them the `rl2` playout net, PR #816, which `0cab615a3`
lacks) all at once, and attributes nothing to any one of them.

| Asc | n | 2026-10-26 | 2026-10-11 | diff pp [paired 95%] | b / c | exact McNemar p |
|---|---|---|---|---|---|---|
| A0 | 200 | 96.0 | 89.5 | +6.5 [+1.6, +11.7] | 19 / 6 | 0.015 |
| A1 | 200 | 99.0 | 95.0 | +4.0 [+0.5, +8.0] | 10 / 2 | 0.039 |
| A2 | 200 | 93.5 | 92.5 | +1.0 [−4.0, +6.0] | 13 / 11 | 0.84 |
| A3 | 200 | 94.0 | 90.0 | +4.0 [−0.9, +9.1] | 16 / 8 | 0.15 |
| A4 | 200 | 95.0 | 86.5 | +8.5 [+3.4, +14.0] | 23 / 6 | 0.0023 |
| A5 | 200 | 91.5 | 85.0 | +6.5 [+1.1, +12.1] | 22 / 9 | 0.029 |
| A6 | 200 | 93.5 | 86.0 | +7.5 [+2.1, +13.1] | 23 / 8 | 0.011 |
| A7 | 200 | 88.5 | 79.0 | +9.5 [+2.9, +16.2] | 33 / 14 | 0.0079 |
| A8 | 200 | 84.5 | 70.0 | +14.5 [+7.2, +21.7] | 44 / 15 | 2.0e-4 |
| A9 | 500 | 56.4 | 39.0 | +17.4 [+12.0, +22.6] | 143 / 56 | 5.8e-10 |
| A10 | 500 | 27.2 | 11.0 | +16.2 [+12.0, +20.4] | 104 / 23 | 1.7e-13 |
| **mean A0–A10** | 2,800 pairs | **83.6** | **74.9** | **+8.7 [+6.6, +10.8]** | | |

b = won on 2026-10-26 and lost on 2026-10-11, c = the reverse. The paired CI
is Newcombe's (1998) square-and-add interval from the two Wilson score
intervals (the analysis files call it "method 10"), with the plain phi
coefficient of the 2×2 table as the correlation term. A variant that
continuity-corrects phi's numerator (AD − BC − N/2, floored at 0, when
AD > BC) moves 8 of the 11 intervals by 0.1pp at one end (A0
[+1.5, +11.8], A8 [+7.1, +21.8]) and changes no interval's relation to zero.
p is the exact two-sided McNemar test, uncorrected for the 11 levels. The mean's CI is a seed-clustered percentile
bootstrap (500 seeds drawn with replacement, each carrying all its levels;
10,000 resamples, numpy `default_rng(20261026)`), because seeds
81,000,000–199 repeat at every level and 81,000,200–499 at A9 and A10; the
stratified bootstrap, an analytic seed-clustered interval and five other RNG
seeds give the same interval. No p-value is given for the mean.

- The new arm is ahead at every level on the point estimate, from +1.0pp
  (A2) to +17.4pp (A9); the paired CI excludes zero at 9 of 11 levels (not
  at A2 and A3). The gains are largest where the old column was lowest, A8
  to A10.
- Run-pooled over all 2,800 pairs (A9 and A10 weigh 2.5×): b/c 450/158,
  +10.4pp.
- The old column, recomputed from its rows (`old_ironclad_runs.csv`) with
  the same code, reproduces the published one exactly: all 11 rates and
  Wilson intervals, the mean 74.9 and the act-1 split. Each seed's act 1 is
  the same in both sweeps (2,800 of 2,800 pairs).

**Next to the §2.6 standard eval.** That eval measured the same champion and
default at another budget and on another panel: Ironclad A10, `a10_it700`
`net_greedy`, 4000/16000 sims, seeds 91,000,000–91,000,399, 107/400 = 26.75%
[22.6–31.3] (`2026-10-26-ic-a10-it700-standard-eval.md`, arm D). This
sweep's A10 is 27.2% (136/500, 8,000 sims). Unpaired, the difference is
+0.45pp, Newcombe hybrid-score 95% CI [−5.4, +6.2]: consistent, and silent on
the effect of budget or panel.

**Post-hoc, not pre-registered: the A9 estimate.** Read off this sweep's A10
runs, the §2.6 "A9 estimate" (won, or reached the second act-3 boss) is
49.8%, against 56.4% for A9 played directly on the same 500 seeds (Ironclad
`a10_it700`, 8,000 sims, #830 default): 6.6pp lower, b/c 87/54, exact
McNemar p = 0.0068, paired 95% CI of direct − estimate [+2.0, +11.2]. The two
are separate plays of each seed, not a replay. The §2.6 write-up already
says the estimate ran about 5pp below a true A9 sweep on Ironclad, so its A9
estimate (52.25%, 4000/16000, another panel) should not be set beside this
sweep's directly played A9.

## Cost

Each run is one single-threaded worker, so per-run wall time stands in for
CPU time.

| Asc (run order) | load at start (1 / 5 / 15 min) | mean s per run | CPU-h | cell wall s |
|---|---|---|---|---|
| A10 | 2.79 / 1.96 / 1.96 | 162.4 | 22.6 | 2,882 |
| A9 | 5.37 / 20.98 / 28.46 | 150.7 | 20.9 | 2,628 |
| A0 | 8.94 / 23.30 / 29.10 | 177.9 | 9.9 | 1,333 |
| A6 | 5.33 / 23.29 / 30.16 | 167.9 | 9.3 | 1,210 |
| A3 | 9.75 / 24.21 / 28.88 | 167.9 | 9.3 | 1,262 |
| A8 | 7.03 / 21.39 / 27.85 | 179.8 | 10.0 | 1,317 |
| A1 | 10.95 / 25.76 / 30.45 | 173.7 | 9.7 | 1,384 |
| A5 | 3.96 / 16.49 / 26.26 | 169.0 | 9.4 | 1,224 |
| A2 | 10.07 / 24.36 / 28.51 | 161.9 | 9.0 | 1,182 |
| A4 | 9.16 / 23.31 / 27.99 | 167.0 | 9.3 | 1,215 |
| A7 | 8.63 / 24.26 / 28.85 | 173.1 | 9.6 | 1,264 |
| **total** | | **165.8** | **128.9** | **4.69 h** |

The host was shared during the sweep: other jobs ran, mostly during A10, A9
and A0 (a docs-site WebAssembly build, an export test with its own wheel
build and test suite, and replays from another reproduction). These are not
clean cost numbers. The win rates do not depend on them: the search has no
wall-clock budget, and the longest run took 490.5 s against the 14,400 s
timeout. The 2026-10-11 Ironclad cells took 223.4 CPU-h (mean 287.2 s per
run, 32 processes); that is context only, on another engine, champion and
search and another day's load.

## Does it still describe main?

The engine source changed after the run, twice by the time of publication.
First PR #941 (co-op AnyAlly cards) and PR #950 (co-op: enemy attacks hit
the teammate, and the search values the teammate's HP); then PR #948
(settle a combat that is already over when it is built) and PR #951
(Whispering Earring turn-1 auto-play), both GH #946, which change
`sts2-core`'s run manager and combat loop and `sts2-mcts`. To check
whether these rows still describe main, the same 12 seeds were replayed
twice, each time on a plain, non-PGO wheel:

- **`b0c3d6d60`**: private main on 2026-10-26 (#944, #936, #941 and #950
  merged since the run), `src_hash` `cb57e2e9bd57b45a`.
- **`78e23253d`**: the merge of this write-up's branch with private main
  `78e23253d` (#945, #948 and #951 merged since `b0c3d6d60`), commit
  `7602c9aaf`, whose engine source is `78e23253d`'s; `src_hash`
  `b53031922847d723`.

Each run was its own one-seed `ppo.probe` (`--seeds 1 --seed-start <seed>
--procs 1`) with the sweep's other flags and checkpoint, 12 at once on a
loaded host. The seeds are the lowest in their level with the outcome
needed:

| level | seed | outcome | final floor | steps | replay on `b0c3d6d60` | replay on `78e23253d` |
|---|---|---|---|---|---|---|
| A10 | 81,000,005 | won | 46 | 623 | identical | identical |
| A10 | 81,000,008 | won | 46 | 784 | identical | identical |
| A10 | 81,000,000 | lost | 45 | 670 | identical | identical |
| A10 | 81,000,001 | lost | 16 | 166 | identical | identical |
| A9 | 81,000,000 | won | 45 | 767 | identical | identical |
| A9 | 81,000,002 | won | 45 | 530 | identical | identical |
| A9 | 81,000,001 | lost | 31 | 391 | identical | identical |
| A9 | 81,000,003 | lost | 19 | 244 | identical | identical |
| A0 | 81,000,000 | won | 45 | 638 | identical | identical |
| A0 | 81,000,011 | lost (one of the 8) | 45 | 688 | identical | identical |
| A7 | 81,000,000 | won | 45 | 593 | identical | identical |
| A7 | 81,000,001 | lost | 45 | 523 | identical | identical |

"Identical" means every field of the row except `wheel`, `py_src` and
`wall_s` (13 fields: the run's settings, `outcome`, `won`, `steps`,
`n_decisions`, `act1`, the final state, `reached_second_boss`, the resolution
counts and `decisions`). `decisions` is empty in every row of the sweep and
of the replays, so the fields that carry play are the other eight from
`outcome` on. The standard eval's `compare_rows.py` agrees on all 24. Each
replay took 1.30–2.13× its published wall time on `b0c3d6d60` and
0.94–1.39× on `78e23253d` (shared host both times). The rows (`so_path`
redacted like the sweep's), the replay logs and the comparisons are in
`replay/<commit>/` (Files); `python replay/compare.py <commit>` reruns a
comparison against `cells/`. So the numbers replay on both commits for those
12 seeds, including the double-boss A10 wins and act-1 and act-2 losses, on
a different build (plain, not PGO). That is evidence, not proof: the other
2,788 runs were not replayed. The claim of record stays "measured on engine
`0181fa552`".

## Caveats

- **Confounded comparison.** The diff against 2026-10-11 changes the
  champion, the search default and 690 commits of engine at once. It says
  the current Ironclad stack wins more on these seeds; it says nothing about
  which change did it.
- **The column table now mixes dates and search stacks.** Wherever the five
  characters are shown together, Ironclad is from this sweep (2026-10-26,
  `a10_it700`, the #830 default, engine `0181fa552`) and Silent, Defect,
  Regent and Necrobinder are from the 2026-10-11 sweep (run 11–13 October,
  engine `0cab615a3`, pre-#830 search, no playout net for any character
  then). Same protocol and seeds, but not the same engine. Since 2026-10-21
  each of those four ships a built-in playout net (GH #822–#825), so their
  rows predate their current shipped search and were not re-run here. The
  one release-protocol number measured since then, in a private check on
  2026-10-21 (`defect-playout-net.md`): Defect `deckmut_it150` (the
  v0.111-vocabulary re-export), `net_greedy`, 8,000 sims, A10 on seeds
  81,000,000–81,000,499 with its playout net, 116/500 = 23.2%, against 22.0%
  (110/500) in the 2026-10-11 column; its heuristic-playout arm on the same
  build read 20.8%. A comparison between the Ironclad row and the other four
  is descriptive.
- **n.** 200 runs per level at A0–A8 give intervals of about ±3–6pp, 500 at
  A9–A10 about ±4–4.5pp (A9 ±4.3, A10 ±3.9). A1 at 99.0% is near the
  ceiling; its interval is asymmetric.
- **One tree.** As on 2026-10-11, the 8,000 sims run as one search tree. The
  live advisor splits that budget into up to 8 root-parallel trees of 1,000;
  one 8,000-sim tree is at least as strong, so these numbers may sit slightly
  above what the advisor plays.
- **The `rc=0` in `progress.log` is not `ppo.probe`'s exit code.** As run,
  `run.sh` wrote `echo "$(date -Is) DONE … rc=$? …"`, and in bash `$?` there
  expands after the `$(date -Is)` substitution, so it records the timestamp's
  status (always 0). The evidence that every cell finished is elsewhere:
  each cell's `.log` ends with its `win rate` line and a passing G1b
  ownership audit, and each `.jsonl` holds its full seed range. The published
  `run.sh` captures the probe's code.
- **Wall times** are from a shared host (Cost).

## Files

In `2026-10-26-ironclad-release-winrates/` (sha256 of each published file in
`SHA256SUMS.txt`; `sha256sum -c SHA256SUMS.txt` from inside the folder):

- `PREREG.md`: the pre-registration, unchanged. It names the private
  worktree's venv the run used.
- `run.sh`: the driver. As run it hard-coded the private paths and refused
  any other `.so`, commit or checkpoint; those are settings now.
- `progress.log`: the driver's log (start, per-level START with the load
  average, DONE with the cell's summary, ALL DONE), unchanged.
- `cells/ironclad_A<asc>.jsonl`: the raw rows, one JSON object per run
  (about 3.0 MB together). `cells/ironclad_A<asc>.log`: `ppo.probe`'s
  summary of each level (outcomes, win rate, wall time, the G1b ownership
  audit), unchanged.
- `old_ironclad_runs.csv`: the 2,800 Ironclad rows of the 2026-10-11 sweep
  that this one pairs with, columns `ascension, seed, act1, won`, in their
  original order. They come from that sweep's per-run file, which stays in
  the private repository (`logs/2026-10-11-release-winrates/runs.csv`,
  sha256 `2c3b2f53…763d`, as `analysis/primary.json` records it), via
  `scripts/extract_old_rows.py`. Their win rates and act-1 split are the
  published 2026-10-11 Ironclad column. The old rows' `total_floor` and
  `wall_s` are left out; every old Ironclad row is `won` or `lost`, so the
  outcome follows from `won`.
- `analysis/scripts/analyze.py` → `analysis/primary.json`,
  `analysis/primary.md`: the pre-registered analysis.
- `analysis/blind/recompute.py` → `analysis/blind/blind.json`: a blind
  recompute from `PREREG.md` and the raw rows.
- `analysis/scripts/reconcile.py` → `analysis/reconcile.json`,
  `analysis/final.txt`: a third recompute that compares every shared value
  across primary, blind, the logs and the published 2026-10-11 column
  (1,925 values, 0 mismatches), and the reconciled numbers and reading.
  `reconcile.json` also records two reporting bugs it found and fixed before
  the final run (`fixes`): the blind's JSON rounded three McNemar p-values
  below 1e-3 to six decimals (A9 and A10 as 0.0), and `primary.md` printed
  "3,400" for the 2,800 runs of the mean row. No result changed; the pre-fix
  outputs it compared against (`analysis/prefix_snapshot/`) stay in the
  private archive.
- `replay/b0c3d6d60/` and `replay/78e23253d/`: the two replays of "Does it
  still describe main?". `rows.jsonl`: the 12 rows in the table's order. As
  run they were 12 one-seed files; they are concatenated here, with only
  `so_path` redacted. `replay.log`: the replay driver's log (the commit, the
  wheel's `src_hash`, the load, each run's finish and exit code), unchanged.
  `compare.txt`: the output of `replay/compare.py <commit>`, the comparison
  with `cells/`. As run, the first replay's comparison read the 12 one-seed
  files from a directory; `compare.py` reads `rows.jsonl` and prints the
  same 12 lines plus a count. The one-seed files, their `ppo.probe` logs and
  the replay drivers stay in the private archive.
- `scripts/redact_paths.py`: the redaction below. `scripts/extract_old_rows.py`:
  builds `old_ironclad_runs.csv` from the private `runs.csv`.

### What changed for publication

The numbers did not change. The run wrote absolute paths of the private host
into its outputs: each row's `wheel.so_path` (also in the replay rows),
and in the analysis files the experiment folder and the private copy of the
2026-10-11 rows.
`scripts/redact_paths.py` replaced the venv part of each path with `<venv>/`,
the folder with `<experiment>` and the old rows' folder with
`<private-repo>/logs/2026-10-11-release-winrates`, and changed no other byte:
every row keeps every field. Hashes the analysis files record (for example
each cell's `jsonl_sha256` in `primary.json`, the inputs of
`reconcile.json`) are the as-run ones, listed here.

The scripts' private paths became arguments, with defaults relative to the
script, and they accept the four-column `old_ironclad_runs.csv`: the old
arm's wall times and floors are then null, and the checks that need them, the
2026-10-11 `progress.log` or `summary.csv` are skipped. `recompute.py` and
`reconcile.py` also got ruff style fixes that change no result. Run on the
as-run inputs at the archive's own path (private rows and old folder, with
the pre-fix snapshot), the published scripts rewrite `primary.json`,
`primary.md`, `blind.json` and `final.txt` byte for byte, and
`reconcile.json` except the two sha256 it records of the current
`analyze.py` and `recompute.py`. `run.sh` keeps the probe command, the level
order and the log format, and now also prints its log lines and records
`ppo.probe`'s exit code.

| file | sha256 as run | sha256 published | change |
|---|---|---|---|
| `cells/ironclad_A0.jsonl` | `5a7de88329cc9e6e2891706191b6482b34384dc3557ab2a24e9b995af37c478e` | `a28d427347565ebe59bddc8a24bbf61609269d2634dec6b40303c4969d025fda` | `so_path` (200 rows) |
| `cells/ironclad_A1.jsonl` | `ecbf7f29e4f02ad0b41da75dafb9e4928ca1b919479da4ed252fce453b6182dc` | `6c2cbb8ceba28573022af9218c4bec7347862849a1399bd32331eeb166dd39c1` | `so_path` (200 rows) |
| `cells/ironclad_A2.jsonl` | `1a2cc7d0d05902d1244ff19ff338c3fb83c494d63b83782b90a68f576dca753a` | `0bcfef02c807ece2d35bb204b6513dedf1aa7fda94f10285af15ac00122a8dff` | `so_path` (200 rows) |
| `cells/ironclad_A3.jsonl` | `dafc433fc1477a1a721b78d4825e4e0f720423ee67baf29689897ab14c75be66` | `3fae7d8f4a80917d5895d6ba0bcdcfb2833acdb1fdc5da47b5da3d4e86cbd834` | `so_path` (200 rows) |
| `cells/ironclad_A4.jsonl` | `c5e4db0c91e1749aa5960e1eb2747dec90dca4a79c41e5598d33aa978f4a8eb1` | `7f6fff1e8dd08be8abcdef4337fd1b72e2a02d5b3a7e1992a4d6c911d245ca91` | `so_path` (200 rows) |
| `cells/ironclad_A5.jsonl` | `c2eb2a605f3b182b73784e07b5a73bf98daea65bfd58a820696114459fb8138c` | `9d0056e33a7b828de4efc89035a9f668216383f1e582502921b41c31ccadbe95` | `so_path` (200 rows) |
| `cells/ironclad_A6.jsonl` | `af6c03878881f5058a846aeaa1cafb4cfa920ba34ea8169e64742e782763aa5d` | `8662241148480e81c37ec63e3af4b283564ac31f1caa5c9b9c6f4f1f98d20692` | `so_path` (200 rows) |
| `cells/ironclad_A7.jsonl` | `09f5d05fdcecf9f09fecef7cadd8b67f2c12c5e2ed240b620af81fd2ad028328` | `c018a29e82fb07c3cda6cd82211ef780e825164637d898ca72b28b98eb68d832` | `so_path` (200 rows) |
| `cells/ironclad_A8.jsonl` | `fe96009b41ddf6565593702e9953ecda6e281ad42fdf4c412922dafd055eac33` | `3662644a2ca863353a302590e44af0ab0dded0304ec5bdba86ed415ba3e8b5a9` | `so_path` (200 rows) |
| `cells/ironclad_A9.jsonl` | `2f8a8e9e593703e4d93d67ec5d0bf719fc4e7709c146b3e299d874c83c65a32b` | `38eb00b3d6a15290e69b3d1e5b28d21f6ea40ca7f4898a5879f91a42c65442a8` | `so_path` (500 rows) |
| `cells/ironclad_A10.jsonl` | `0cacffcc0dc1c9ef5fb1e209d3bb771a6a4575f769fcbdcbfaafe50d7146da6c` | `58ad1278e20f107e21d333b9b9644a15922b944cdb86da22a9c86979a630a643` | `so_path` (500 rows) |
| `analysis/primary.json` | `f39b0cbddc670cf95b5850ce366d1bcdf405a93b7cf173f63a53f6f6b0158647` | `507c74e3fc65660f49e44b7f056433fa5ba69d2224efd9c4bfa4cdd26f20d19b` | three paths |
| `analysis/blind/blind.json` | `27903205017cc43a9117ab5f9c1823be529f20f09419d2ce3ccde6d4284e0051` | `78e7ed3e77a0c5377f719062465eb03637b14f93bd89418a9977826f9398e9e6` | four paths |
| `analysis/reconcile.json` | `c2aa2d396ad4c25e2caca135377fa210edd0989b260b1a9a659c58f9317c7118` | `dc0ed6d583328adc383a0ecdfb85115b4d12e80e697ce3745b73e654bb83a8c8` | five `so_path` |
| `run.sh` | `dee9b63fb8d08cbcc6b7f410a83f42cda1998fb2202109038d8c4f9a86a324ac` | `d06f9a032751bd3345ca1f98b7888c5cf4156e71cf2a2b2a28516c4628f83dbb` | paths and pins → settings; logs to the terminal; the probe's exit code |
| `analysis/scripts/analyze.py` | `743f1730c12c5741aa665255918015693650e702830a0c4a965265fcc5783ba9` | `1a95e46e750da2640c58ce144c4f3d8499f01e25aba0a280d72a72755290e1a7` | old rows → argument; four-column old rows |
| `analysis/blind/recompute.py` | `5dc7a14df8b2902c8136e4cc890d604edc0739d506e8fb4f6a8a904ce49b02d3` | `ab6e99a821f66b037334a818871404d055d9accaccd545f501081191cda750a1` | folder and old rows → arguments; four-column old rows; style |
| `analysis/scripts/reconcile.py` | `677e19f1fd78d7e711da53b646570652c77ffaa05dc9afae3fc27f406d7045f3` | `23282829d59b57313c0e0bc8bd715efe7e66435d1e1eac2bb6ff0ea42553d2f4` | folder and old rows → arguments; skips; style |
| `replay/b0c3d6d60/rows.jsonl` | `a4ef78fa3dfdaade700ea789b4d006cee148cfb7f4d2e1cb992ee791438266ed` (the 12 one-seed files concatenated) | `35831a2d6a4ec904d90e5c1ec899eea642a5bbb7cbceaa8c2e03d707b7c08d99` | `so_path` (12 rows) |
| `replay/78e23253d/rows.jsonl` | `27ddfde6841cc60914af239ae38d989fa87529b95a5a8d67df622a79cc8b1f00` (the 12 one-seed files concatenated) | `547de4afe7adeeb7f7d96dd072a95eaec87a0f53f40edb520fe6d3dbccca7726` | `so_path` (12 rows) |

`PREREG.md`, `progress.log`, the 11 cell `.log` files, `analysis/primary.md`,
`analysis/final.txt`, `replay/compare.py` and each replay's `replay.log`
and `compare.txt` are byte-identical to the private archive's.

### What a rerun on this folder reproduces

Reproduce step 5 runs the three analysis scripts on a copy of this folder
alone. Every rate, interval, paired test and bootstrap comes out the same:
`reconcile.json`'s `reconciled` block, which holds all of them, is identical,
and its recompute finds 0 mismatches. The files differ from the published
ones in these fields only, because the folder does not carry the old rows'
`total_floor` and `wall_s`, the 2026-10-11 `progress.log` and `summary.csv`,
or the pre-fix snapshot, and because the redaction changed the rows' bytes:

- `primary.json`, 72 values. `integrity.new_per_level.A*.jsonl_sha256` (11):
  the hashes of the published rows instead of the as-run rows (both in the
  table above). `inputs.old_runs_csv` and `inputs.old_runs_csv_sha256`: the
  published CSV instead of the private `runs.csv`. Null instead of a number:
  `cost.per_level.A*.old_wall_s_mean`, `old_wall_s_median`,
  `old_cpu_hours`, `old_batch_wall_s` and `wall_s_mean_ratio_new_over_old`
  (55), `cost.total.old_cpu_hours` and `old_batch_wall_h`, and
  `depth_old.per_level` (with its `field` text saying why).
- `primary.md`, 25 lines: the integrity line names `old_ironclad_runs.csv`,
  and the old columns of the cost table (12 rows) and of the run-depth table
  (11 rows, plus its "Old:" sentence) read `n/a`.
- `blind.json`, 14 values: the old rows' path in `meta.inputs`, and null for
  `cost_wall_s.A*.old_arm` (11), `cost_overall.old_mean_s` and
  `cost_overall.old_sum_h`.
- `reconcile.json`: 1,668 values compared instead of 1,925, 0 mismatches.
  The 257 it skips check the old arm's wall times and floors against
  `primary.json` (112) and `blind.json` (57), and the old column against the
  2026-10-11 `progress.log` (44) and `summary.csv` (44). With no pre-fix
  snapshot, `fixes` is empty and `prefix_snapshot` and `verdict.prefix_*`
  are null, so `verdict.statement` is shorter. `inputs` holds the hashes of
  the files it read (15 differ).
- `final.txt`, four places: the verdict paragraph (1,668 values, no pre-fix
  record), the old rows' file name, the clause about `summary.csv`, and the
  old arm's wall-time line.

This was checked on 2026-10-26 on a copy of the folder as committed.

## Reproduce

On Linux (the driver reads `/proc/loadavg` and calls `sha256sum`). The
standard eval's Reproduce section lists the prerequisites (git, Rust 1.96 via
`rustup`, `uv`; about 6 GB of disk). No game files are needed: `ppo.probe`
runs the Rust engine only.

**0. Check out the run's engine, with this folder.** The public repository
moves with every sync, and a later engine change can change the rows
(step 4). The run's engine is the one in the public commit that first
carried the standard eval write-up; this folder came later, so take it from
the commit that first carried this write-up:

```bash
git clone https://github.com/tfriedel/spire-mentor && cd spire-mentor
NEW=$(git log --diff-filter=A --format=%H -1 -- docs/experiments/2026-10-26-ironclad-release-winrates.md)
ENG=$(git log --diff-filter=A --format=%H -1 -- docs/experiments/2026-10-26-ic-a10-it700-standard-eval.md)
git checkout "$ENG"
git checkout "$NEW" -- docs/experiments/2026-10-26-ironclad-release-winrates.md docs/experiments/2026-10-26-ironclad-release-winrates
```

The second checkout only adds documents; it does not touch the engine source.

**1. Build the wheel and rebuild the checkpoint.** Follow steps 1 and 2 of
the Reproduce section of `2026-10-26-ic-a10-it700-standard-eval.md`: `uv sync
--dev` in `sts2-pycli/` (the wheel must report `src_hash` `6a98001ae326637b`
and carry the 268,757-byte built-in prior), then its `onnx_to_ckpt.py`
writes `a10_it700_from_onnx.pt`. PGO (`just sim-engine-pgo`) is optional; it
only makes runs about 1.3× faster. From here on every command runs from
`sts2-pycli/`.

**2. Run the sweep.** 2,800 runs, 128.9 CPU-hours on the PGO wheel the run
used (shared host); on the plain wheel of step 1 expect about 1.3× that,
about 170 CPU-hours or 6 h on 30 processes. Per level, on the PGO wheel: A9
and A10 about 21–23 CPU-hours each, A0–A8 about 9–10 each (Cost). The driver
runs the levels in the run's order, writes `cells/` and `progress.log` like
the published folder, resumes a level whose log is complete, and prints its
log lines; `wc -l ic-release/cells/*.jsonl` shows progress within a level.

```bash
R=../docs/experiments/2026-10-26-ironclad-release-winrates
CKPT=a10_it700_from_onnx.pt OUT=ic-release PROCS=30 bash $R/run.sh
```

It scrubs every `STS2_*` variable and runs one `ppo.probe` per level, which
you can also run by hand (`ASC_ORDER="10 9"` limits the driver to some
levels):

```bash
for v in $(env | grep '^STS2_' | cut -d= -f1); do unset "$v"; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python
mkdir -p ic-release/cells
for asc in 10 9 0 6 3 8 1 5 2 4 7; do
  n=200; [ "$asc" -ge 9 ] && n=500
  .venv/bin/python -m ppo.probe --policy net_greedy --ckpt a10_it700_from_onnx.pt --character Ironclad \
    --ascension "$asc" --act1 roll --seeds "$n" --seed-start 81000000 --procs 30 \
    --sims 8000 --boss-sims 8000 --run-timeout 14400 --out "ic-release/cells/ironclad_A$asc.jsonl"
done
```

`--procs` sets only how many runs play at once; each run depends on its seed
alone. `ppo.probe` takes one contiguous seed range, so a spot check is one
call per seed, e.g. `--ascension 10 --seeds 1 --seed-start 81000005`.

**3. Score.** `ppo.probe` prints each level's win rate. The analysis scripts
in the folder are written for this sweep's rows: they check its build pins
(the private PGO wheel), so on your own rows their integrity block reports
your wheel and they exit 1.

**4. Pair with the published rows.**

```bash
C=../docs/experiments/2026-10-26-ic-a10-it700-standard-eval/scripts/compare_rows.py
for asc in 0 1 2 3 4 5 6 7 8 9 10; do
  .venv/bin/python $C $R/cells/ironclad_A$asc.jsonl ic-release/cells/ironclad_A$asc.jsonl
done
```

`compare_rows.py` compares what play determines (outcome, steps, decisions,
final state, `reached_second_boss`, resolution counts) on the seeds both
files hold, and ignores wall time and the build fields. If it reports
differences, your engine source differs from the run's; pair your own runs
on that engine instead of comparing with these rates.

**5. Rerun the analysis on the published folder.** The scripts need numpy and
scipy (both in the `uv sync --dev` environment). Run them on a copy named
`<experiment>`, passed as that relative path: `primary.json` records the
folder as given, and the published file says `<experiment>`. Run them in this
order, since `reconcile.py` reads the other two outputs.

```bash
REPO=$(git rev-parse --show-toplevel); R=$REPO/docs/experiments/2026-10-26-ironclad-release-winrates
cd "$(mktemp -d)" && cp -r "$R" '<experiment>'
for s in analysis/scripts/analyze.py analysis/blind/recompute.py analysis/scripts/reconcile.py; do
  "$REPO/sts2-pycli/.venv/bin/python" "<experiment>/$s" '<experiment>' > /dev/null
done
diff -rq "$R" '<experiment>'
```

`diff` lists the five analysis files; "What a rerun on this folder
reproduces" above says exactly which fields differ and why.
