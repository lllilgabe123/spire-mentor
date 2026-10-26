# Ironclad `a10_it700` standard strength eval on the GH #830 default (2026-10-26)

**Status: done (2026-10-26).** 800 runs, 0 errored. This is the playbook §2.6
measurement (`docs/ppo-champion-training-playbook.md`). It is descriptive and
not a gate: it changes no decision. The Ironclad search default rests on the
two A10 gates at 1000/4000 (below).

`PR #…` and `GH #…` numbers and commit hashes refer to the private development
repository. Where this write-up relies on a private record, it states the
fact here. Everything needed to rerun and re-score the eval is public: the
sibling folder `2026-10-26-ic-a10-it700-standard-eval/` holds the per-seed
rows, the driver and the analysis scripts, and **Reproduce** below runs it from
a clone of the public commit that first carried this write-up.

## Purpose

PR #942 (GH #830, 2026-10-25) made the forms + combo knobs and the BC move
prior the Ironclad run agent's default. From then on, the §2.6 command
measures a setting that no reference row in the playbook had run: those rows
ran the knobs and no prior. PR #942 asked for `a10_it700` to be re-measured on
the ship wheel before a new Ironclad net is compared with the table. This run
does that (arm D). It adds a same-wheel control with the prior switched off
(arm P): the setting of the old row, run on this wheel. The analysis was fixed
in `PREREG.md` before the runs.

## Setup

| | |
|---|---|
| Code | private main `a1bd07c71` (the PR #942 merge `118085173` plus a docs merge). Its engine source (`sts2-rs/`) is unchanged up to private `059a49d83`, and the Python changes up to there are in the co-op advisor (`combat_mirror`, `coop_coordination`, `realgame_advisor*`), which `ppo.probe` does not import. The first engine change after the run is PR #941 (co-op AnyAlly cards, merged 2026-10-26) |
| Wheel | PGO build, `src_hash` `4e5bb730d28e2976`, `.so` sha256 `823fa393…3ba2`, `py_src` `774d91045cdb5433` (the private build; a public build of the same source reports another `src_hash`, see Reproduce) |
| Built-in prior | `sts2-rs/data/prior_nets/ironclad.json`, compiled in: `builtin_prior_net_json("Ironclad")` sha256 `ca469259…d4f2`, 268,757 bytes |
| Run policy | `a10_it700`, `net_greedy`. The run loaded the torch checkpoint `ppo_fullrun_ironclad_a10_it700.pt` (sha256 `91c83c1a…44fc`), which the public snapshot does not ship; its weights ship as `sts2-pycli/data/ppo/ppo_fullrun_ironclad_a10_it700.onnx` (sha256 `0b091ad6…8de4`), and `scripts/onnx_to_ckpt.py` rebuilds the checkpoint from it |
| Combat search | honest BeliefDraw, Ironclad `rl2` playout net (`sts2-rs/data/rollout_nets/ironclad.json`, compiled in), 4000 sims / 16000 at bosses; knobs `removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0` (the env line of §2.6, also the default) |
| Panel | Ironclad A10, seeds 91,000,000–91,000,399 (400), `--act1 roll`: 189 Overgrowth, 211 Underdocks, the same per seed in both arms |
| Driver | `ppo.probe --policy net_greedy --character Ironclad --ascension 10 --act1 roll --seeds 400 --seed-start 91000000 --procs 30 --sims 4000 --boss-sims 16000 --run-timeout 14400` |
| Env | every other `STS2_*` unset; BLAS threads 1 |
| Host | a shared 48-CPU Linux machine, 30 procs; arms in sequence, D first |

Arms:

- **D, the shipped default.** The §2.6 command verbatim. The run agent plays
  the knobs in every fight and the built-in BC move prior in elite and boss
  fights (PUCT, mix 0.5, T 1).
- **P, prior off.** D plus `STS2_RUNAI_PRIOR_NET=off`: the knobs only. This is
  the setting of the 2026-10-23 playbook row, on this wheel, which includes
  PR #940's rule 5c (the junk-dispose root guard).

## Results

Both arms: 400 rows, 0 error, 0 timeout, exit 0, exactly the 400 panel seeds
with no duplicates. Paired n = 400, no seed dropped.

| arm | A10 wins | A10 [Wilson 95%] | A9 estimate | A9 estimate [Wilson 95%] |
|---|---|---|---|---|
| D, default (prior on) | 107/400 | 26.75% [22.65, 31.29] | 209/400 | 52.25% [47.36, 57.10] |
| P, prior off | 96/400 | 24.00% [20.07, 28.42] | 199/400 | 49.75% [44.88, 54.63] |

All four numbers: Ironclad A10, n = 400 paired seeds, `a10_it700` `net_greedy`,
`rl2` playout net, honest BeliefDraw at 4000/16000, the shipped knobs, PGO
wheel `4e5bb730d28e2976`; D has the BC prior on in elite and boss fights, P
has it off. The A9 estimate counts a run that won or entered the act-3 second
boss room (`reached_second_boss`). `a10_score.py` prints the same rates
rounded to one decimal (26.8% / 52.2% and 24.0% / 49.8%;
`analysis/a10_score.txt`).

D is the new playbook §2.6 row; P is listed under it as the same-wheel
pre-ship setting.

## Paired D − P (descriptive)

| | both won | D only | P only | neither | D − P | exact McNemar p | 95% CI |
|---|---|---|---|---|---|---|---|
| A10 | 59 | 48 | 37 | 256 | +2.75pp | 0.278 | [−1.76, +7.26]pp |
| A9 estimate | 154 | 55 | 45 | 146 | +2.50pp | 0.368 | [−2.38, +7.36]pp |

McNemar is the exact binomial on b of b + c at 1/2, two-sided. The CI is
Newcombe (1998) method 10 with the uncorrected φ = (ad − bc)/√(efgh), where
e, f, g, h are the four margins. The continuity-corrected φ = (|ad − bc| −
n/2)/√(efgh) gives [−1.79, +7.29] and [−2.41, +7.38]; these two intervals are
computed from the rows, and no analysis file carries them. A paired Wald CI
gives [−1.76, +7.26] and [−2.39, +7.39]. Both intervals include zero, so at
n = 400 this panel cannot tell the prior's effect from no effect. D − P is
also a different contrast from the gates (see below).

Where act-3 losses ended: D lost 84 runs in act 3 before the second boss, P 82.
Of those, 68 (D) and 70 (P) ended on act floor 15, the first boss floor; 16 and
12 ended earlier in act 3.

## Cost

Each run is one single-threaded worker, so wall time per run stands in for CPU.

| | D | P | D / P |
|---|---|---|---|
| arm wall | 2258 s (37.6 min) | 2149 s (35.8 min) | 1.051 |
| per run, mean | 160.5 s | 152.4 s | 1.053 |
| per run, median | 150.2 s | 144.7 s | 1.038 |
| per run, max | 623.7 s | 523.8 s | |

Adjusted for run length: 1.039 per step (ratio of sums), 1.043 (ratio of the
arms' mean per-run seconds per step), 1.040 per combat decision, and a geomean
of 1.037 over the 231 paired seeds that ended on the same floor. The paired
per-seed D/P ratio of seconds per step, (D `wall_s`/`steps`) / (P
`wall_s`/`steps`), has median 1.025 and geomean 1.045 (range 0.31–5.94); it
is computed from the rows. The analysis files' paired per-seed ratios
(median 1.038, geomean 1.068) are the same ratio for per-run wall time, not
per step. So D ran about 4% slower per step than P. The arms ran one
after the other on a shared host (1-min load 6.83 at D's start, 7.83 at P's),
so part of that may be host load.

No prior-only cost at 1000/4000 exists to compare with. At 1000/4000 the
private identity check of the ship (the record that PR #942 shipped exactly the
gated search) measured the whole bundle, knobs + prior against neither, at
1.002–1.027× per combat step, 1.014 pooled over 44 runs on this `.so`; the gate
measured its bundle-against-plain-search ratio at 1.042.

## How this relates to the old rows and the gates

- **The 2026-10-23 row** (the same §2.6 command on the same panel: Ironclad
  A10, n = 400, `a10_it700` `net_greedy`, 4000/16000, the knobs, no prior:
  23.0% A10 / 49.8% A9 estimate) ran on another host and an older wheel
  without rule 5c. Its per-seed rows are not on this host, so it is quoted,
  not paired. P is the same setting on this wheel and is consistent with it:
  1.0pp above it at A10 (24.00% vs 23.0%) and level with it on the A9 estimate
  to the printed decimal (49.75% vs 49.8%). The comparison is unpaired across
  hosts and wheels, and the paired D − P standard error on this panel is
  already 2.3pp, so it cannot show that P reproduces the old row.
- **D is 3.75pp above that row at A10.** That gap is unpaired and crosses
  wheels and hosts. It mixes the prior with the other wheel changes and with
  run-level noise, so it is not a measurement of the prior.
- **The `plr_it550` row** (17.0% A10 / 45.2% A9 estimate, 2026-10-21; the
  same command and panel: Ironclad A10, n = 400, `net_greedy`, 4000/16000, the
  knobs, no prior) is another net on an older wheel. It stays in the table as
  history.
- **The gates** measured a different contrast at a different budget: the
  bundle (knobs plus prior) against plain search, Ironclad A10 at 1000/4000,
  `a10_it700` `net_greedy`. PR #938: +5.20pp, 1,500 paired, p 1.5e-5, repeated
  97.06% CI [+2.62, +7.78]. A second, separately pre-registered gate on another
  machine: +3.45pp, 4,000 paired, p 8.8e-07. D − P here is the prior alone on
  top of the knobs, at 4× the budget. Neither gate number is a direct reference
  for +2.75pp. Both gate statements are quoted in
  `sts2-rs/data/prior_nets/README.md`.
- **The prior's own share** at the gated budget is still unmeasured
  (`sts2-rs/data/prior_nets/README.md`: neither gate had a prior-only arm).
  D − P is a prior-only contrast, but at 4000/16000 and with an interval from
  −1.76 to +7.26pp it does not settle that.
- Only the run agent at 1000/4000 was gated; the 4000/16000 standard eval is
  one of the budgets the gates do not cover. This run does not fill that gap.
  It measured the prior alone, and D − P is positive with an interval that
  includes zero. The gated contrast, the bundle against plain search, is still
  unmeasured at 4000/16000.

## What this does not re-measure

PR #942 added dated notes to Ironclad numbers measured before the default
changed. Only the playbook §2.6 note was about this protocol; it is replaced
by the new rows. The others qualify numbers from other protocols and stay:

| note | the number it qualifies | protocol that would re-measure it |
|---|---|---|
| `docs/experiments/2026-10-11-release-winrates.md`, `docs/site/index.html`, the public `README.md` (`release/public-overlay/README.md` in the private tree) | the release Ironclad column (`plr_it550`, A0–A10) | the release sweep: `ppo.probe` `net_greedy`, 8,000 sims in one tree incl. bosses, seeds 81,000,000+, 200 per cell at A0–A8 and 500 at A9–A10, on the current champion and the default |
| `sts2-pycli/data/ppo/ppo_fullrun_ironclad_a10_it700.provenance.md` | the promotion gates (A10 × 1000, A9 × 400, pooled A0–A10 × 200) | the §2.1–2.3 promotion gate at 1000/4000; the note already says how to reproduce the old search |
| `sts2-pycli/python/ppo/combat_bench/` (`SCHEMA.md`, `agents.py`, `devpanel.py`) | B0 on the combat bench (`pro_v1`, `dev_v1`) | a B0 re-anchor with `ppo.combat_bench` on the ship wheel (`dev_v1` reps 0–39) |
| `sts2-rs/data/prior_nets/README.md` | the prior's own share of the gated gain | knobs-only and prior-only arms against control at 1000/4000, A10, fresh seeds (about 1,500–3,000 pairs per arm) |
| playbook §1 (collection) | the bundle's cost, ~1.02–1.04× per combat step at 1000/4000, and the ungated bundle at 250/1000 | a cost probe and a screen at 250/1000 |

PR #942 also asked for a decision on the published win-rate pages. That
decision is still open: this run does not settle it. The pages
(`docs/site/index.html`, the public `README.md`,
`docs/experiments/2026-10-11-release-winrates.md`) keep their "measured
before" note until then. The protocol that would settle it is the release
sweep at 8,000 sims (first row above) on `a10_it700` with the default.

> **Update, later on 2026-10-26: that sweep has run**
> (`2026-10-26-ironclad-release-winrates.md`: Ironclad `a10_it700`
> `net_greedy`, 8,000 sims in one tree, the #830 default, engine `0181fa552`,
> seeds 81,000,000+; A10 27.2% and A9 56.4% on 500 seeds each, mean A0–A10
> 83.6%). The three pages now carry its Ironclad row in place of the
> "measured before" note, and the first row of the table above no longer
> applies.

## Caveats

- **The arm labels rest on `scripts/run_eval.sh`.** No row, stdout or stderr
  line records the prior or the env. `driver.log` logs the env before
  `env STS2_RUNAI_PRIOR_NET=off` is applied, so its P line does not show the
  variable. The evidence that the switch acted is indirect but strong: 399 of
  400 seeds differ between the arms (398 in step count, 379 in final state,
  169 in final floor, 144 in final HP, 399 in resolution counts), and 85
  differ in A10 outcome. Runs replay seed-for-seed on this `.so` at 1000/4000:
  the private ship identity check passed 51 of 51 checks with 0 differences,
  over runs that include 4 in arm P's exact setting (the knobs with
  `STS2_RUNAI_PRIOR_NET=off`) and 50 with the prior off in all. The run itself
  did not check replay at 4000/16000; within it, only the one identical seed,
  91,000,303, is consistent with it. That seed is an 84-step act-1 death on
  floor 8 (66 combat decisions; the rows do not record room types), which can
  end before any elite or boss fight. D's ~4% extra time per step points the
  same way, which is weak evidence on its own. Later checks are direct: at
  4000/16000 a public build replayed 24 runs on 14 seeds exactly, the default
  reproducing D's rows and `STS2_RUNAI_PRIOR_NET=off` reproducing P's
  (Reproduce, step 5). That confirms the labels and the determinism at this
  budget, across two builds, on those 14 seeds; it does not cover the other
  386.
- **n = 400 at 4× the deployed budget.** The interval on D − P is about
  ±4.5pp at A10 and ±4.9pp on the A9 estimate.
- **The A9 estimate is not an A9 sweep.** On Ironclad it ran ~5pp below a
  true A9 sweep (2026-08-24, n = 200; `a10_score.py`, playbook §2.6).
- **Host load.** The arms were not interleaved, so the cost ratio carries
  host noise. The win rates do not depend on wall time: the search has no
  wall-clock budget, and no run came near the 14,400 s timeout (max 624 s).

## Files

In `2026-10-26-ic-a10-it700-standard-eval/` (sha256 of each published file in
`SHA256SUMS.txt`; `sha256sum -c SHA256SUMS.txt` from inside the folder):

- `PREREG.md`: the pre-registration, unchanged. Its Build lines name the
  private worktree, venv and build commit the run used.
- `scripts/run_eval.sh`: the driver. As run it hard-coded the private paths and
  refused any other `.so` or commit; those are now settings (see below).
- `scripts/analyze.py` → `analysis/primary.json`: the primary analysis.
- `scripts/recompute_blind.py` → `analysis/recompute_blind.json`: a blind
  recompute from the raw rows only.
- `scripts/reconcile.py` → `analysis/reconcile.json`: a third recompute from
  the raw rows, compared with both (62 values, 0 mismatches), and the written
  reconciliation of the two (its `reconciliation` block).
- `analysis/final.txt`: the reconciled numbers and reading. Its wording was
  revised after review of PR #943 (cost labels, the old-row comparison, the
  replay condition); no number changed. It cites the scripts at
  `analysis/scripts/`, where the private archive kept them. It and
  `reconcile.json` also cite the "ship identity record"
  (`docs/experiments/2026-10-25-ic-bundle-ship-identity.md`), a private record
  that the public snapshot does not carry; the facts used from it are stated
  under Caveats and Cost.
- `analysis/a10_score.txt`: `docs/experiments/ironclad-a10/a10_score.py` on
  each arm and paired (P as champion, D as candidate).
- `logs/driver.log`, `logs/D.stdout`, `logs/P.stdout`: the `ppo.probe`
  summaries (G1b ownership audit PASS in both). Both stderr files were empty.
- `runs/D.jsonl`, `runs/P.jsonl`, `runs/*.exit`: the raw rows (about 0.9 MB
  together), one JSON object per run.
- `scripts/onnx_to_ckpt.py`: rebuilds the `a10_it700` torch checkpoint from the
  shipped ONNX (Reproduce).
- `scripts/compare_rows.py`: checks whether a rerun replays the published rows,
  seed by seed.
- `scripts/redact_paths.py`: the redaction below.
- `SHA256SUMS.original.txt`: the checksums of the private archive before the
  redaction below (so `final.txt` there already has its post-review wording).

### What changed for publication

The numbers did not change. The run wrote two absolute paths of the private
host into its outputs: each row's `wheel.so_path` (the engine `.so` in the
private venv) and the experiment folder (`prereg` in `primary.json`;
`experiment` and `inputs` in `recompute_blind.json`). `scripts/redact_paths.py`
replaced the venv part with `<venv>/` and the folder with `<experiment>`, and
changed no other byte: every row keeps every field, so a `so_path` now reads
`<venv>/lib/python3.13/site-packages/sts2_rs/sts2_rs.abi3.so`. The scripts'
hard-coded private paths became arguments, with defaults relative to the
script; their analysis is unchanged. `recompute_blind.py` and `reconcile.py`
looked for the folder two levels up, where the archive kept them
(`analysis/scripts/`); here it is one level up.

A reproduction from a public export then changed the scripts once more, so
that a rerun on someone else's rows works and does not carry this run's text:

- `run_eval.sh` also creates `analysis/`, prints its log lines to the
  terminal, and records `head=none` cleanly outside a git checkout.
- `analyze.py`, `recompute_blind.py` and `reconcile.py` create `analysis/`
  when it is missing.
- `analyze.py` writes its fixed qualifier text (the private checkpoint
  `91c83c1a` and PGO wheel `4e5bb730d28e2976`) and the host sentence of
  `cost.notes` only when `runs/D.jsonl` and `runs/P.jsonl` are this run's rows
  (their sha256 as run or as published). On other rows the qualifier names
  the wheel the rows record and says that the rows do not record the
  checkpoint.
- `reconcile.py` writes its `reconciliation` block, the written verdict and
  evidence checks of this run, under the same condition, and a note
  otherwise.
- `recompute_blind.py` got three ruff style fixes, which change no result, and
  `docs/experiments/ironclad-a10/a10_score.py` names its file encoding.

| file | sha256 as run | sha256 published | change |
|---|---|---|---|
| `runs/D.jsonl` | `6e15ab979b8171c03be324d6a56e17d2fcee9f85c34f0849dbfe4229f100420c` | `e732981cd9c0585d1a4a4184a92f071375f2b8addd71386ef5e423390caa2903` | `so_path` redacted (400 rows) |
| `runs/P.jsonl` | `f74084528d78718b262ffba9b0edf25b0a6189fc5deebe0d1063a94920c94f63` | `8dad9c0cfbcf3cb3feac5430b8e322acdb12be901e8d5c1549853cd6b4583f4f` | `so_path` redacted (400 rows) |
| `analysis/primary.json` | `ed5a121c41d1f9909f150858ebee737a13d3f64b8015a4f857f738c7ed946809` | `6481c5f6d84392b2cad6c60741b39eb092446667260c82c7b31ef640ab77a36c` | `prereg` path |
| `analysis/recompute_blind.json` | `63491be65be87c6f4f4769c17c231e32b7bdd6a841f4744f8b47202bc0e787fe` | `dc5d5af2f8c01d8197e2d60f5d3db26f1413a383f6dbc83de7c44830d4d4ae4f` | folder paths, two `so_path` |
| `analysis/reconcile.json` | `038fc846082e7c5703e29892e9876a41100857753349156d27ec8c3830e657aa` | `f8faa72ccd932f36d2d2670eefb1759978074039c16c9f81f39574db964c6a7d` | two `so_path` |
| `scripts/run_eval.sh` | `7ef747170796087db07de7e460ab471c1dc553f61d66cfd66877df8368f3508b` | `495cb9875df61b60463889dee2ea3b1ff398863bd6843752bb699dc5359fb2bd` | paths and pins → settings; creates `analysis/`, logs to the terminal |
| `scripts/analyze.py` | `72d1d8cb3dc990c4579d2afaca8fa3a39ef3221305c1e86f4cddbfcd3d1fa9ea` | `cc266b2b6d30f5a764edb2f255dd8d82294f44b4cc8174ace6af1fa039a18b9e` | folder → argument; fixed text only on this run's rows; creates `analysis/` |
| `scripts/recompute_blind.py` | `584a2d12e763de79f3c914e36793ff9dfd64048c0919b9b011080b8136ca8a82` | `13a617687468e6aa3c34e4e7e06c5b7e072e06a24144a07c196ba0941aeba292` | default folder one level up; creates `analysis/`; style |
| `scripts/reconcile.py` | `be52c45f63bc3daa11d09411e7a452b460487e3ba7276a482de5dd516b2d033f` | `534751c1ddb11097021cb644511a00b4a0457d2dcd34968dd54af32dd3330c2a` | folder → argument; written reconciliation only on this run's rows; creates `analysis/` |

`PREREG.md`, `analysis/a10_score.txt`, `analysis/final.txt`, the three logs
and the two `.exit` files are byte-identical to the private archive's. Checked
with the published scripts: `a10_score.py` on the published rows prints
`analysis/a10_score.txt` byte for byte, and `analyze.py`,
`recompute_blind.py` and `reconcile.py` run on a copy of this folder named
`<experiment>` (passed as that relative path, with scipy installed) rewrite
the three analysis files byte for byte. On a driver-shaped folder with no
`analysis/` and the same rows marked as from a plain public wheel, they write
the same numbers, name that wheel in the qualifiers, leave out the private
host sentence and the written reconciliation, and report 0 mismatches.

## Reproduce

On Linux (checked; macOS is untested, and the driver reads `/proc/loadavg`
and calls `sha256sum`). You need git, Rust 1.96 (`rustup` installs it from
`sts2-rs/rust-toolchain.toml`) and `uv`, which fetches Python 3.13 if it is
missing. The README also lists cargo-nextest and `just` for the engine's own
tests; this eval needs neither, except `just` for the optional PGO build. No
game files are needed: `ppo.probe` runs the Rust engine only. Budget about
6 GB of disk (the `.venv` is 5.4 GB) and about 45 CPU-hours for both arms.

**0. Check out the commit that first carried this write-up.** The public
repository moves with every sync, and a later engine change can change the
rows (step 5).

```bash
git clone https://github.com/tfriedel/spire-mentor && cd spire-mentor
git checkout "$(git log --diff-filter=A --format=%H -1 -- docs/experiments/2026-10-26-ic-a10-it700-standard-eval.md)"
```

From step 1's `cd sts2-pycli` on, every command runs from `sts2-pycli/`.

**1. Build the engine wheel** (about 2–3 min of compiling on 12 cores, plus
the downloads).

```bash
cd sts2-pycli
uv sync --dev     # Python deps (torch, onnx, scipy) and a plain build of the sts2_rs wheel
.venv/bin/python -c 'import sts2_rs; print(sts2_rs.src_hash(), len(sts2_rs.builtin_prior_net_json("Ironclad") or ""))'
```

The second number must be 268757: the wheel carries the built-in prior, i.e.
the GH #830 default. The first is the engine's source hash. The run's engine
source, as the public snapshot carries it, hashes to `6a98001ae326637b`: a
wheel built from an export of private main `0181fa552` plus this write-up
reported it, and the engine's own hasher (`sts2-rs/crates/sts2-srchash`) gives
it for the export based on `059a49d83`, whose engine source is the same. The
private build of the same source reports `4e5bb730d28e2976`: `src_hash` hashes
the source bytes, and the export strips quoted C# from comments in the engine
crates. If yours reports another hash, your engine source differs from the
run's (for example by PR #941, the first engine change after the run), and
step 5 shows whether the rows still replay.

PGO is optional. `just sim-engine-pgo` (bash, about 15–18 min) builds the PGO
wheel the run used; it only makes runs about 1.3× faster. On the run's engine
source the PGO build's own identity check (`sts2-rs/scripts/pgo_wheel.sh`)
found the PGO wheel decision-identical to the plain build over 50 full runs
(10 seeds × 5 characters at 250/1000 sims), and the private ship identity check
found all 16 of its row files identical between the PGO wheel and a plain
wheel of the same source at 1000/4000.

**2. Rebuild the run policy's checkpoint.** `ppo.probe --ckpt` loads a torch
checkpoint; the public snapshot ships the champion as ONNX only.

```bash
E=../docs/experiments/2026-10-26-ic-a10-it700-standard-eval
.venv/bin/python $E/scripts/onnx_to_ckpt.py --out a10_it700_from_onnx.pt
```

The script checks the ONNX's sha256, copies its weights into a torch state
dict and refuses to write it unless the tensors hash to the private
checkpoint's (digest `fa6148d6…cf24`). It holds 52 of the checkpoint's 54
tensors bit for bit; the two it lacks belong to an auxiliary training head
whose output `ppo.probe` never reads. Loaded into `NetPolicy`, the rebuilt and
the private checkpoint differ only in that head, so they make the same
decisions.

**3. Run both arms.** Each arm is 400 runs. A run took about 155 s on the PGO
wheel the run used; on the plain wheel of step 1 expect about 1.3× that
(1.2–1.9× on a loaded host). Both arms together are about 45 CPU-hours, about
1.5 h on 30 processes. The driver runs D, then P, writes `runs/`, `logs/` and
`analysis/` like the published folder, and prints its log lines;
`wc -l ic-a10-eval/runs/*.jsonl` shows progress within an arm.

```bash
CKPT=a10_it700_from_onnx.pt OUT=ic-a10-eval PROCS=30 bash $E/scripts/run_eval.sh
```

It scrubs every `STS2_*` variable and then runs these two commands, which you
can also run by hand:

```bash
for v in $(env | grep '^STS2_' | cut -d= -f1); do unset "$v"; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python
export STS2_RUNAI_MCTS_OVERRIDES="removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0"
ARGS="--ckpt a10_it700_from_onnx.pt --policy net_greedy --character Ironclad --ascension 10 --act1 roll
      --seeds 400 --seed-start 91000000 --procs 30 --sims 4000 --boss-sims 16000 --run-timeout 14400"
mkdir -p ic-a10-eval/runs
.venv/bin/python -m ppo.probe $ARGS --out ic-a10-eval/runs/D.jsonl                            # arm D, the default
STS2_RUNAI_PRIOR_NET=off .venv/bin/python -m ppo.probe $ARGS --out ic-a10-eval/runs/P.jsonl   # arm P, prior off
```

`--procs` sets only how many runs play at once; each run depends on its seed
alone. `ppo.probe` takes one contiguous seed range, so a spot check of a few
seeds is one call per range (e.g. `--seeds 1 --seed-start 91000085`).

**4. Score.**

```bash
R=ic-a10-eval/runs
.venv/bin/python ../docs/experiments/ironclad-a10/a10_score.py $R/D.jsonl
.venv/bin/python ../docs/experiments/ironclad-a10/a10_score.py $R/P.jsonl
.venv/bin/python ../docs/experiments/ironclad-a10/a10_score.py $R/P.jsonl $R/D.jsonl   # paired D - P
# the full analysis into ic-a10-eval/analysis/ (in this order: reconcile.py reads the other two)
for s in analyze recompute_blind reconcile; do .venv/bin/python $E/scripts/$s.py ic-a10-eval; done
```

The full analysis needs the driver's folder: `analyze.py` reads
`logs/driver.log` for the arm wall times. On your rows, the qualifiers in
`primary.json` name your wheel, and `reconcile.json` carries a note where the
published one has this run's written reconciliation; its
`comparison_mismatches` is the check (empty: the three recomputes agree).

**5. Pair with the published rows.**

```bash
.venv/bin/python $E/scripts/compare_rows.py $E/runs/D.jsonl $R/D.jsonl   # did D replay?
.venv/bin/python $E/scripts/compare_rows.py $E/runs/P.jsonl $R/P.jsonl
.venv/bin/python ../docs/experiments/ironclad-a10/a10_score.py $E/runs/D.jsonl $R/D.jsonl   # paired outcomes
```

`compare_rows.py` compares what play determines (outcome, steps, decisions,
final state, `reached_second_boss`, resolution counts) on the seeds both files
hold, and ignores wall time and the build fields. Checked on 2026-10-26 on an
export of the run's engine source (a plain `uv sync --dev` wheel, `src_hash`
`6a98001ae326637b`, the checkpoint rebuilt by `onnx_to_ckpt.py`, 8 processes
on a loaded host): 24 runs on 14 seeds at 4000/16000 replayed the published
rows exactly.

- Seeds 91,000,212–91,000,219 in both arms: 8 of 8 in D and 8 of 8 in P. They
  include D's wins on 91,000,213 and 91,000,219, which P lost, and P's win on
  91,000,216, which D lost. The arms took 286 s and 319 s for the 8 runs.
- Six seeds chosen from the published rows, one run each. In D: 91,000,085
  (D's shortest loss, in act 1), 91,000,264 (D's longest act-3 loss),
  91,000,147 and 91,000,269 (won in both arms), 91,000,035 (D won, P lost) and
  91,000,188 (P won, D lost). In P: 91,000,035 and 91,000,188. 6 of 6 in D and
  2 of 2 in P; each run took 1.2–1.9× its published wall time.

On an engine whose source differs from the run's, the rows may differ. If
`compare_rows.py` reports differences, pair your own D and P on that engine
instead of comparing with these rates (playbook §2.6).
