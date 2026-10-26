# site-bench

The non-Rust harnesses behind the performance numbers on the docs site and in
the READMEs, and (under `bench/`) the drivers that ran the 2026-10 benchmark.
The method, the exact commands, the results and the caveats are in
[`sts2-rs/docs/perf/2026-10-14-benchmarks.md`](../../sts2-rs/docs/perf/2026-10-14-benchmarks.md),
which supersedes
[`2026-10-14-site-performance.md`](../../sts2-rs/docs/perf/2026-10-14-site-performance.md).
The Rust harnesses are cargo examples:
`sts2-rs/crates/sts2-bench/examples/site_random_bench.rs` (random-play fights,
workload W1), `sts2-rs/crates/sts2-bench/examples/xsim_toadpoles.rs` (sansb's
Toadpole fixture, workload W2) and
`sts2-rs/crates/sts2-workbench/examples/hint_bench.rs` (the browser hint's
search, native).

| File | Measures |
|---|---|
| `cs_random_fights.py` | the same random fights through the shipped game, run headless, one fresh game process per fight (cold JIT every fight; needs the game's assemblies and a Release build of `sts2-pycli/src/Sts2Headless`) |
| `cs_fights_pool.py` | the same fights again, but MANY fights per game process (warm JIT), on one core or on the whole machine. `prep` builds one save per seed in fresh processes (untimed); `pool` starts N workers on their CPU sets (`--workers`, `--cpu-sets`, or `--pin core\|thread\|none`), releases them together and collects. Each fight is reset from its seed's save (untimed) and checked against the fresh process's start state; `--ref-jsonl` compares every fight's length and outcome with `cs_random_fights.py` rows. Reports wall time, process CPU and game-thread CPU per action, the JIT warm-up curve and RSS |
| `scale_runner.py` | whole-machine scaling of any single-threaded harness: N copies, each pinned before exec, released at the same instant. Presets from the topology in `/sys`: `1core` (one CPU, SMT sibling idle), `1core-smt` (both threads of that core), `ccx` / `ccx-smt` (one L3 domain), `physN` (N cores, one thread each, spread over the L3 domains), `smt48` (every hardware thread). Prints per-process and aggregate rates and the scaling efficiency against a 1-process `--baseline`; `--perf` adds clock and IPC per child, `--identity` checks that the children did the same work, `--parser hint_bench` reads `hint_bench`'s text output |
| `train_iter_bench.py` | one real PPO training iteration (`ppo.trainer`, the deployed recipe, warm-started from a champion checkpoint) on the whole machine. Checks a fingerprint of iteration 0's games before comparing timings and proves from `/proc/<pid>/maps` which `sts2_rs` extension every worker loaded |
| `wasm_hint_bench.mjs` | the browser hint's search as WASM in Node |
| `chrome_bench.html`, `chrome_bench_page.js`, `chrome_driver.mjs` | the same search as WASM in headless Chromium, via Playwright |
| `advisor_bench.py` | the advisor's search (`sts2_rs.mcts_recommend` with the advisor's settings) on a fixed Act-1 boss position. `--pin-workers CPULIST` pins the search's worker threads one per CPU after the warm-up (off by default; see "CPU masks" below) |
| `ppo_collect.py` | one PPO collection worker (`ppo.workers.play_batch`) on fixed seeds |
| `cpu_busy.py` | the quiet-host check run before every metric |
| `xsim/sansb/`, `xsim/sts2core/` | our cross-engine harnesses for the two external Rust engines, [sansb/sts2-sim](https://github.com/sansb/sts2-sim) and [lain-wood/sts2core](https://github.com/lain-wood/sts2core): W1 and W2 written against each engine's public API, dropped into its checkout and built with its own release profile. Their READMEs give the commits, the build, the workload differences that cannot be removed and the `--policy` choice |
| `bench/` | the drivers that built, measured and analysed the 2026-10 benchmark (next section) |

Every harness prints enough to check that two engines played or searched
identically (a digest, a position fingerprint and first move, or a per-run
hash); compare those before comparing timings.

**Linux only.** The harnesses pin with `taskset` and read `/proc`
(`cpu_busy.py`, `cs_random_fights.py`, `cs_fights_pool.py`, `scale_runner.py`,
`train_iter_bench.py`); `advisor_bench.py` records the process's CPU affinity
where the OS reports one.

**CPU masks for multi-threaded cells.** Give a multi-threaded search a mask
made of whole L3 domains (`lscpu -e` or
`/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`), or pin each
worker to its own CPU. Linux places a woken thread on an idle CPU inside its
own L3 domain only, so a mask that straddles two domains (6 CPUs of one, 2 of
the next) keeps stacking two workers on one CPU while another sits idle. On the
24-core EPYC behind the published numbers, `taskset -c 6-13` did exactly that
to the 8-thread advisor (21.7% of repetitions; 0.8% with `6-17`). The 2026-10
benchmark therefore measures the 8-thread advisor on two whole L3 domains
(`6-17`) and with `--pin-workers 6-13`, never on a bare `6-13` mask.

## `bench/`: the 2026-10 benchmark drivers

| Path | Does |
|---|---|
| `bench/env.sh` | sourced by every driver: `REPO` (the checkout; default the one holding this file), `BENCH_DIR` (scratch root, required), `SB` (this directory) |
| `bench/measure_all.sh` | the serial measurement window: runs the blocks below in order, each sub-driver gating on a quiet host; resumable (`runs/done/<block>`) |
| `bench/ours/` | our engine, previous (`d3ea66a51`) against new (`6d1918237`): `export.sh` (git archive of both trees), `build.sh` (the native harnesses and the plain `sts2_rs` extension for both, with byte-identical harness files), `smoke.sh` (untimed identity checks), `measure_ours.sh` (W1, W2, hint, advisor and the whole-machine scale runs) |
| `bench/xsim/` | the external engines: `build_ext.sh` (clone, check out and build both with our harnesses), `measure_xsim.sh` (W1, W2, each engine's own benchmark, whole-machine scale runs, our new engine interleaved in every round), `perenc_compare.py` (per-encounter win rate and actions per fight across the three engines, untimed) |
| `bench/csharp/` | the shipped game, headless: `build_cs.sh` (Release `Sts2Headless` plus the trees the harnesses expect), `measure_cs.sh` (cells A0, A, B2-B16, C1-C6), `m2_reference.jsonl` (the published `cs_random_fights.py` run's per-fight length and outcome, which every cell is checked against) |
| `bench/training/` | training speed: `export.sh`, `build.sh` (plain extensions for both engines and a frozen copy of the PGO wheel), `measure.sh` (`percore`: one pinned collection worker; `full`: whole trainer iterations), `summarize.py` |
| `bench/training/diag/` | the watchdog-free replays of iteration 0 behind the "typical iteration" figures: `run.sh pool\|replay\|trace ARM TAG` drives `poolrep.py` (the 960 games through a pool shaped like the trainer's, 46 workers, no stall watchdog), `replay.py` (one process per game) and `trace_game.py` (per-step times of one game); `run_all_pools.sh` runs the three full pools one after another |
| `bench/wasm/` | the browser hint search as WASM: `build_wasm.sh` (the shipped wasm recipe for both engines), `measure_wasm.sh` (Node and headless Chromium), `summarize_wasm.py` |
| `bench/facts/` | `hw_snapshot.sh` (hardware and toolchain dump; its last line is JSON) and `hardware.template.md` (the hand-written host table the analysis reads) |
| `bench/results/analyze.py` | the analysis of the raw data: writes `$BENCH_DIR/results/results.json` and `RESULTS.md` |
| `bench/results/apply_xcheck.py` | run right after `analyze.py`: re-derives the headline numbers from the raw files with its own parsing, and adds to `results.json` the training typical-iteration figures and the previous engine's caveat (from `training/diag/`'s replay) and the cross-engine whole-machine ratios. The published numbers come from the two scripts together; `apply_xcheck.py` changes `results.json` only |
| `bench/w1_encounters.txt` | the 22 Act-1 Overgrowth encounters of W1, in our registry order |

### Reproducing the whole benchmark

**What it needs.**

- A Linux host with `perf` (user-mode counters), `taskset`, Python 3 (the
  drivers use the standard library only), rustup and git. The CPU numbers in
  the drivers are the published host's: "one core" is CPU 20 with its SMT
  sibling 44 idle, the L3 domains are 0-5, 6-11, 12-17 and 18-23 (siblings
  +24), and the whole machine is 0-47. On another machine set `CORE`,
  `CORE_SET`, `CPUS` and the advisor masks in `ours/measure_ours.sh` to match.
- A **quiet host.** Every timed stage first waits (up to `WAIT_MAX_MIN`, 30
  min) for a 1-minute load average under 1.5, the whole host under 3% busy
  and, for per-core stages, the stage's own CPUs and their SMT siblings under
  5% busy (the training stages gate on the whole-host busy share only). A
  stage that never sees a quiet host is skipped, not run; `FORCE=1` runs it
  anyway and records the flag. Nothing else may run during the window,
  including builds and the C# `prep`.
- About 13 GB free under `BENCH_DIR`.
- Toolchains: Rust 1.99.1 (pinned by `sts2-rs/rust-toolchain.toml`; sts2core is
  built with it too), rustc 1.97.1 for sansb/sts2-sim (pinned by its own
  `rust-toolchain.toml`, installed by rustup on first use), a .NET 9 SDK for
  the C# block, Node with `npm ci` done in `sts2-pycli/web` (Playwright and
  `wasm-opt`) and Playwright's Chromium installed (`npx playwright install
  chromium`; `PW_ROOT` and `WASM_OPT` override where the drivers look),
  the `wasm-bindgen` CLI at the version in `sts2-rs/Cargo.lock` (0.2.120), and
  for training the checkout's `sts2-pycli/.venv` with torch and CUDA (`PY`)
  plus one idle GPU (`GPU`, default index 1).
- **The game's assemblies, for the C# block only.** `csharp/build_cs.sh`
  copies `sts2.dll` and its dependencies from `GAME_DIR` (default
  `sts2-pycli/lib`, which `just import-game` fills from your own copy of the
  game). The published run used `sts2.dll` md5
  `282bfb4ccec5b17c51dea6b37ccb4791` (game build v0.107.1). The Rust, WASM,
  cross-engine and training blocks do not need the game.
- **The external engines, for the cross-engine blocks only.** `xsim/build_ext.sh`
  clones them into `SANSB_SRC` and `STS2CORE_SRC` (defaults
  `$BENCH_DIR/ext/sts2-sim` and `$BENCH_DIR/ext/sts2core`) and checks out
  the measured commits: sansb/sts2-sim `c61336cf05d289ae01ab9254beaa62c1b4ee407c`
  (game v0.111.0) and lain-wood/sts2core
  `3b2d969d025b9e64bc96b9a15928bd2a0d824bf2`.
- **The PGO wheel, for training only.** `training/build.sh` freezes a copy of
  the `sts2_rs` wheel installed in the checkout's venv as the `new-pgo`
  configuration; build it first with `just sim-engine-pgo` in `sts2-pycli/`.
  The champion checkpoint (`ppo_fullrun_ironclad_plr_it550.pt`) is taken from
  the new commit's tree and must be a real blob, not an LFS pointer.

**Setup (untimed; it saturates the machine, so finish it before the window).**

```bash
export BENCH_DIR=/path/to/scratch          # required by every driver
B=scripts/site-bench/bench
bash $B/ours/export.sh                     # src-base (d3ea66a51), src-new (6d1918237)
bash $B/ours/build.sh                      # both trees in parallel, 5-11 min here
bash $B/ours/smoke.sh                      # identity checks, base == new; must end "0 failed"
bash $B/wasm/build_wasm.sh                 # needs ours/export.sh
bash $B/xsim/build_ext.sh                  # clones + builds sansb/sts2-sim and sts2core
bash $B/csharp/build_cs.sh                 # needs GAME_DIR
bash $B/csharp/measure_cs.sh prep          # 8,448 per-seed saves, about 10 min on 40 processes
bash $B/training/export.sh
bash $B/training/build.sh                  # needs the PGO wheel in the checkout venv
mkdir -p "$BENCH_DIR/facts" && cp $B/facts/hardware.template.md "$BENCH_DIR/facts/hardware.md"   # fill it in by hand
```

`BASE_REF` / `NEW_REF` (export scripts) measure other commits. The binaries
behind the published numbers were built before a comment-only edit to
`xsim_toadpoles.rs` (its header now says which sansb random player it
matches): a rebuild from this tree gives that file a different `HARNESS_MD5`
(published: `2726e738108968c14d69d09eff41efdd`) and may move panic-location
line numbers in the binary, but the code is the same.

**The measurement window.**

```bash
bash $B/measure_all.sh                     # all blocks in order; re-run to resume
# training's "typical iteration": watchdog-free replays of iteration 0 (the
# published ones ran right after the window, one at a time, on a quiet host)
bash $B/training/diag/run_all_pools.sh                                   # pool-{pgo,new,base}-full
bash $B/training/diag/run.sh replay base a4 --asc 4                      # the previous engine's 85 A4 games
bash $B/training/diag/run.sh replay new s703 --seeds 10000703 --procs 1  # game 703 on its own
bash $B/training/diag/run.sh replay pgo s703 --seeds 10000703 --procs 1
bash $B/training/diag/run.sh trace base s703 --seed 10000703 --asc 4     # per-step times of game 703
bash $B/training/diag/run.sh trace pgo s703 --seed 10000703 --asc 4
python3 -I $B/results/analyze.py "$BENCH_DIR"
python3 -I $B/results/apply_xcheck.py "$BENCH_DIR"   # after analyze.py, every time
```

`measure_all.sh` runs these blocks in this order; a block can also be run on
its own (`bash $B/measure_all.sh wasm`). The times are the published run's
on the 24-core host (2026-10-14, 10:59-16:33 UTC, about 5.5 h of wall time
including the gate waits):

| Block | Runs | Published wall time |
|---|---|---|
| `provenance` | `facts/hw_snapshot.sh`, `PROVENANCE.env` of ours and of the external engines | 3 s |
| `ours_core` | `measure_ours.sh w1 hint adv w2`: one core, previous against new in ABBA pairs | 26 min |
| `xsim_core` | `measure_xsim.sh w1 w2 own`: one core, every engine and policy in rotated rounds, plus each external engine's own benchmark | 14 min |
| `perenc` | our per-encounter W1 baseline and `perenc_compare.py` (untimed same-work check) | 4 s |
| `scale_rust` | `measure_ours.sh scale_w1 scale_w2 scale_hint scale_w1_base` and `measure_xsim.sh scale`: the whole machine, 3 repeats per preset, each with a fresh 1-process baseline | 94 min |
| `wasm` | `measure_wasm.sh node chrome` | 12 min |
| `cs` | `measure_cs.sh A0 A`, then `B2 B4 B8 B16 C1 C2 C3 C4 C5 C6` with 220 warm-up fights per process | 20 min (A0 to B16) + 59 min (C1 to C6, about 47 min of it waiting at the gate) |
| `training` | `training/measure.sh percore 3` (one pinned collection worker, 3 rounds of the three extensions in alternating order), then `full 2` (whole trainer iterations, 2 repeats of the 3 configurations, ABBA) | 84 min |
| `closing` | the end-of-window hardware snapshot, the training and WASM summaries | 3 s |

**The C# whole-machine gate.** `measure_cs.sh`'s strict gate also requires
every CPU of a cell, and its SMT sibling, to be at most `CPU_BUSY_MAX` (5)
percent busy over 2 s. For the 48-CPU C cells that never passed on the shared
host: one briefly busy CPU anywhere fails it. In the published run the `cs`
block was stopped while C1 waited, and C1-C5 were re-run after the training
block, with C6 (384 processes) added then, gated on the 1-minute load and the
whole-host busy share only. That is the same gate the Rust whole-machine
scale runs use. It is now the default for the C cells
(`C_CPU_BUSY_MAX=100`); `C_CPU_BUSY_MAX=5` restores the strict gate. The
per-core cells (A0, A, B) keep the strict gate on CPUs 20 and 44.

**Outputs.** Everything lands under `BENCH_DIR`: `runs/` (all Rust, WASM and
cross-engine runs, `all.log`), `csharp/runs/` (`results.jsonl`, one line per
cell), `training/results/`, `training/runs/` and `training/diag/out/` (the
replays), `facts/` and `results/`. `analyze.py` reads those (except the
replays, which `apply_xcheck.py` reads), `facts/hardware.md` and
`csharp/m2_reference.jsonl`.
Some of its prose and checks describe the published run (its timeline, the C6
cell, its known caveats); check them before publishing numbers from another
run.
