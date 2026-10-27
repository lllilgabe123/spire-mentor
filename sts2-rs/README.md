# sts2-rs

Deterministic Rust simulator of **Slay the Spire 2** combat and full runs
(pinned to game build **v0.107.1**), built for fast MCTS rollouts and RL
training.

- **Game-behavior source of truth:** the C# decompilation at
  `../sts2-wiki/decompiled/v0.107.1/` (a sibling in the monorepo — useful for
  verifying mechanics, **not** required to build or test this workspace).
- **Port provenance:** the engine scaffolding was originally ported from an
  archived Python sim; see [`PYTHON_SIM_PIN.md`](PYTHON_SIM_PIN.md) for what the
  `// Ported from sts2-rl-agent@…` headers mean.
- **What is and is not redistributable:** see [`../PROVENANCE.md`](../PROVENANCE.md).

## Quickstart (standalone)

No sibling projects are needed to build and test:

```sh
cargo check --workspace --all-targets
cargo nextest run --workspace     # canonical test runner (see note)
cargo test --workspace --doc      # doctests — nextest cannot run these
```

The `sts2-py` crate (PyO3) needs **Python ≥ 3.11** at build and test time. If
the `python` on your PATH is older or missing (a `pyo3-build-config` build
script failure), set `PYO3_PYTHON` to a 3.11+ interpreter (e.g. the output of
`uv python find --system 3.13`), or skip the crate with `--exclude sts2-py`. Windows
has an extra DLL trap: see
[`../BUILDING-WINDOWS.md`](../BUILDING-WINDOWS.md#4-the-sts2-py-pyo3-crate-needs-a-real-python).

The toolchain is pinned by [`rust-toolchain.toml`](rust-toolchain.toml) to
**Rust 1.99.1**; the MSRV is **1.99** (`rust-version` in
[`Cargo.toml`](Cargo.toml)). All workspace crates share version **0.0.1**
(there is no published release).

> **Use `cargo nextest`, not `cargo test`.** Tests run one-per-process under
> nextest, which isolates the process-global effect/relic registry — plain
> `cargo test` shares one process and produces order-dependent false failures.
> Slow tests (name prefix `slow_`) are excluded from the default run; execute
> them with `cargo nextest run --profile slow`. Install with
> `cargo install cargo-nextest --locked`. The one correct `cargo test`
> invocation is `cargo test --workspace --doc`, because nextest cannot run
> doctests at all.

## Crates

Nine of the ten workspace members live under `crates/` (the tenth,
`xtask/`, is unshipped repo dev tooling — `cargo run -p xtask -- <cmd>`):

| Crate | What it is |
|---|---|
| `sts2-core` | Deterministic combat engine: state, hooks, commands, RNG, damage, powers. |
| `sts2-content` | Card / power / monster / relic / potion / orb definitions for all five characters. |
| `sts2-mcts` | UCT MCTS search over `sts2-core` states. |
| `sts2-runai` | Run-level AI: arena harness, seed panels, baseline policies, PPO policy/value nets (ONNX inference), MCTS combat bridge. |
| `sts2-cli` | Debug REPL + scripted-replay driver. Main binaries: `sts2-cli`, `sts2-corpus-gen`, `sts2-fight-replay`, `sts2-replay-audit`, `run_arena`; further auto-discovered dataset-extraction bins live in `src/bin/`. |
| `sts2-wire` | PyO3-free wire-format parse + `CombatState` reconstruction (the hint/bridge dialect used by the real-game bridge and advisor). |
| `sts2-py` | PyO3 bindings — builds the `sts2_rs` Python extension module. |
| `sts2-bench` | Criterion benchmarks (`transitions_per_sec`, `mcts_smoke`, …). |
| `sts2-srchash` | Dependency-free content hashing of a crate's build inputs; shared by engine `build.rs` scripts and a runtime recompute so the parity harness can prove a loaded extension matches its source. |

### Dependency sketch

Arrows point from dependent to dependency; `sts2-srchash` is a
`build.rs` dependency of the engine crates. Note `sts2-mcts` is
deliberately content-agnostic: it depends only on `sts2-core`
(`sts2-content` appears only in its dev-dependencies).

```text
Engine stack:

  sts2-runai ─► sts2-mcts ─► sts2-core
  sts2-runai ─► sts2-content ─► sts2-core

Consumers (direct dependencies):

  sts2-wire  ─► sts2-core, sts2-content, sts2-runai
  sts2-cli   ─► sts2-core, sts2-content, sts2-mcts, sts2-runai
  sts2-bench ─► sts2-core, sts2-content, sts2-mcts, sts2-cli
  sts2-py    ─► sts2-core, sts2-content, sts2-mcts, sts2-wire,
                sts2-runai, sts2-cli, sts2-srchash
```

In words: `sts2-core` is the foundation; `sts2-content` layers game content on
it; `sts2-mcts` searches over both; `sts2-runai` drives full runs on top of the
search; `sts2-wire`, `sts2-cli`, `sts2-bench`, and `sts2-py` are consumers that
each pull in the slice they need.

Besides `crates/`, the workspace carries `tests/parity/` (fixture store for the
parity harnesses — see its README), `fuzz/` (a deliberately separate cargo-fuzz
workspace; needs nightly to *run*, stable to `cargo check`), and `docs/`
(plans, audits, perf notes — largely lab-notebook history).

## `data/` directories

| Directory | Contents |
|---|---|
| `data/seed-panels/` | Fixed seed-panel JSON files consumed by the `sts2-runai` arena harness — pinned seed sets so win-rate evaluations are reproducible and comparable across policies. |
| `data/run-corpus/` | Vendored corpus of real human `.run` files (git-lfs, `files.tar.zst` + `unpack.sh`) used by the replay/parity harnesses. Game-derived data — see `data/run-corpus/README.md` and [`../PROVENANCE.md`](../PROVENANCE.md). |
| `data/runai/` | Serialized nets + `nets/manifest.json` provenance ledger for the run-AI. Only `value_human_a0.json` remains deployable (`STS2_RUNAI_VALUE` + `--policy value-net`); the ranker pipeline was removed 2026-07-12 and its files are historical. `nets/recipe/` scripts are historical provenance, not runnable. |
| `data/run-trajectories/` | Recorded run trajectories (e.g. `ppo-ironclad-a0`) from training/eval sweeps. |
| `data/value_policy_net/` | Value/policy-net experiment artifacts (embeddings, configs). |

The **PPO run-policy champions** served per character are managed from the
sibling `../sts2-pycli` project: `data/ppo/champions.toml` there is the single
source of truth, and the Rust-side resolver match arms in
`crates/sts2-runai/src/ppo/net.rs` are **generated** from it by
`sync_champions.py` (`just check-champions` gates the sync). Do not hand-edit
the generated blocks. See `../docs/ppo-champion-training-playbook.md`.

## CLI

`sts2-cli` is the debug REPL / scripted-replay driver, subcommand-shaped:

```sh
cargo run -p sts2-cli -- combat --seed 0 --encounter nibbit
cargo run -p sts2-cli -- combat --agent random --seed 0 --steps 1000 --jsonl /tmp/out.jsonl
cargo run -p sts2-cli -- list relics
```

Every starter surface — deck, relics, potion belt, encounter group — is
selectable via inline flags (`--deck Strike,Strike,Bash+`), presets
(`--deck-preset`, `--relic-preset`, `--potion-preset`), external files, or a
single bundled TOML scenario (`--scenario path.toml`; examples in
`docs/scenarios/`, dump the resolved form with `--print-scenario`). Card /
relic / potion names are accepted in both `PascalCase` and display-string
form, case-insensitively. During interactive play, `hint [SIMS]` runs
`sts2-mcts` on the current state and prints the top root actions.

`sts2-cli --help` and `sts2-cli combat --help` list the full flag surface;
`list <kind>` accepts `encounters`, `cards`, `relics`, `potions`, or `content`
for a cross-cutting implemented-surface inventory (`--all` dumps full enums).

## Python extension

`sts2-py` builds the `sts2_rs` Python module via PyO3/maturin, installed into
the consuming venv:

```sh
uv pip install maturin
cd crates/sts2-py
maturin develop --release
python -c "import sts2_rs; print(sts2_rs.version())"
```

There is no PyPI release — the consumer is the sibling `../sts2-pycli` project,
which wires the dependency via `maturin develop` (or a portable wheel from
`maturin build --release`). See
[`crates/sts2-py/README.md`](crates/sts2-py/README.md) for the full build path
and rationale.

## Performance

Measured 14 October 2026 on an AMD EPYC 7443P (24 cores / 48 threads, Linux
6.8), plain `--release` (fat LTO, no PGO); medians of interleaved runs on a
quiet host. One core = one pinned process with its second hardware thread
idle; whole machine = one pinned process per hardware thread (48), started
together by `scripts/site-bench/scale_runner.py`. All results, the hardware,
the shipped game and two other STS2 simulators:
[benchmarks page](https://tfriedel.github.io/spire-mentor/benchmarks.html);
method and commands:
[`docs/perf/2026-10-14-benchmarks.md`](docs/perf/2026-10-14-benchmarks.md).

| Metric | Harness | One core | Whole machine |
|---|---|---|---|
| Random-play fights/s (Ironclad starter deck + Burning Blood vs each of the 22 Act-1 Overgrowth encounters in turn) | `cargo run --release -p sts2-bench --example site_random_bench` | ~59 k | ~1.4 M |
| Engine actions/s (same run; an action is a card play or an end of turn) | same | ~1.3 M (~0.77 µs each) | ~31 M |
| The browser hint's search, 1,000 sims, one thread per search (seed BENCH1, first fight) | `cargo run --release -p sts2-workbench --example hint_bench` | ~14 ms (~73 searches/s) | ~1,490 searches/s |

`transitions_per_sec` (criterion) is an EndTurn-only regression canary with
no cards, relics or monster AI, not a throughput figure. The `sts2_rs` Python
extension links mimalloc; the benchmark binaries above use the system
allocator.

## Testing & parity

The full validation-gate list, beyond the quickstart:

```sh
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo nextest run --workspace          # never bare `cargo test`: shared-process false failures
cargo test --workspace --doc           # the one `cargo test`: nextest cannot run doctests
cargo check --target wasm32-unknown-unknown -p sts2-workbench-wasm   # the browser build
cargo check --manifest-path fuzz/Cargo.toml   # after changing CombatState / Relic shapes
```

`--profile release-checked` keeps release optimisation but turns overflow
checks and debug assertions on, for sweeps that should panic on silent
arithmetic bugs. [`tests/parity/README.md`](tests/parity/README.md) describes
the parity fixture store: frozen Python-sim-era fixtures plus the live
C#-oracle-era artifacts (the oracle harness itself is driven from
`../sts2-pycli`).

## CI

This directory has no workflows of its own. In the public repository, the
root `.github/workflows/ci.yml` runs the game-free gates above on every push
and pull request. The private development repository runs the same gates
locally only.

## Status

All five characters (Ironclad, Silent, Defect, Necrobinder, Regent) are
ported and checked against the C# oracle. In the last full sweeps Ironclad was
clean on all 6,104 fuzzed fights (June 2026), the other four on 99.4–99.8%
(July 2026, a known tail still open), and 1,232 of 1,250 whole runs were clean
(July 2026); see `docs/audits/2026-07-07-parity-sweep-findings.md`. Full-run
PPO champions are trained and served for every character.
[`CHANGELOG.md`](CHANGELOG.md) records the phase-by-phase history (its
"0.1.0" entry is a combat-engine milestone label from the porting plan, not
a released version — the workspace version is 0.0.1). Historical phase
snapshots live under `docs/audits/` and `docs/reality-checks/`.
