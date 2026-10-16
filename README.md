# spire-mentor

[![CI](https://github.com/tfriedel/spire-mentor/actions/workflows/ci.yml/badge.svg)](https://github.com/tfriedel/spire-mentor/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tfriedel.github.io%2Fspire--mentor-blue)](https://tfriedel.github.io/spire-mentor/)

**A bit-exact re-implementation of Slay the Spire 2, an AI that plays it, and
the tools to watch, question and argue with that AI — in a browser tab or
inside the real game.**

![The browser workbench in the act-3 boss fight: the best card carries a 95% win badge and the machine-reasoning panel ranks every legal move](docs/site/img/wb-combat.webp)

<sub>The browser workbench in the act-3 boss fight: the search marks the best
card with its win chance, and the machine-reasoning panel ranks every legal
move.</sub>

## What's here

- **A from-scratch Rust engine for the whole game.** All five characters,
  ~580 cards, ~300 relics, the potions, powers, monsters, events and all
  three acts, pinned to game build v0.107.1 — down to a hand port of .NET's
  `System.Random`, so the same seed produces the same map, the same
  rewards and the same shuffles as the real game.
- **Proven against the real game code, not eyeballed.** A headless harness
  runs the game's own C# assembly in lockstep with the Rust engine and
  compares the full state after every action, and a fuzzer hunts for the
  cases nobody thought to test. In the last full sweeps Ironclad was clean
  on all 6,104 fuzzed fights (June 2026), the other four characters on
  99.4–99.8% (July 2026), and 1,232 of 1,250 whole runs were clean (July
  2026). No full sweep has run since.
- **Fast.** ~59k random-play fights/s (~1.3M engine actions/s) on one
  core, ~31M actions/s on a 24-core server. With both kept busy that is
  about **1,500× the shipped game's throughput per core** (run headless, on
  the same fights) and 2,430× on the whole machine. The browser hint's
  1,000-simulation MCTS search takes ~14 ms native, ~18 ms as WASM in
  Chromium. Speed never trades away accuracy: the parity harness above
  holds at full speed. All numbers, the hardware and a comparison with two
  other STS2 simulators:
  [benchmarks](https://tfriedel.github.io/spire-mentor/benchmarks.html).
- **Usable from Python.** PyO3 bindings (`sts2_rs`) drive combats, MCTS
  searches and whole runs from Python, and the PPO training stack is built
  on them.
- **An AI that wins.** Chance-node MCTS for every fight, plus a PPO-trained
  transformer per character for every other decision: paths, card picks,
  shops, events, rest sites, removals and upgrades. The search is
  non-clairvoyant: draws and enemy rolls are chance nodes, so it never sees
  the draw order. In the simulator Ironclad wins **90%** of runs at A0,
  **39%** at A9 and **11%** at A10, and Defect wins **22%** at A10 (every
  character and level is in the table below). In the **real game**, playing
  unassisted at **Ascension 8**, the August 2026 versions won 13 of 40
  Ironclad runs and 11 of 40 Silent runs.
  Explained in depth: [how the AI plays](https://tfriedel.github.io/spire-mentor/ai.html)
  and [how it's trained](https://tfriedel.github.io/spire-mentor/training.html).
- **A browser workbench.** The game, with its graphics and sound, compiled
  to WebAssembly: it runs in any browser on any platform, offline, with no
  server. Rewind to any step and branch off to play a situation several
  ways; let the AI coach you; play it as an autobattler where you make the
  run decisions and the AI fights; review a finished run for its biggest
  blunders; copy a position from the real game in to analyse it; explore
  the MCTS search tree. The art and sound come from your own copy of the
  game (`just import-game`).
- **SimLab.** The workbench's "what if" lab for a single fight: set up any
  combat (character, deck, HP, relics, potions, enemies, ascension) and see
  the legal moves and what the search thinks of each. Replay the fight many
  times to see the spread of outcomes, and test a card, a relic or a line of
  play without reaching that spot in a real run.
- **The Spire Mentor mod for the real game.** Brings many of the workbench's
  features into the game itself: an in-game coach (ranked moves, card
  scores, route advice, win chance), an autopilot that plays whole runs, a
  bit-exact rewind button, and a link that opens the current game state in
  the workbench.
- **Bring-your-own-game tooling.** One command, `just import-game`,
  decompiles your copy of the game and extracts its art, audio and data; the
  same pipeline can generate a full static wiki.

## Numbers

| | |
|---|---|
| Engine coverage | 5 characters · 580 card ids · 297 relics · 63 potions · 272 powers · 121 monsters · 69 events · 4 act models |
| Parity vs the game's C# | lockstep after every action, a 5,000-seed fuzz tier · last full sweeps: Ironclad 6,104/6,104 fuzzed fights clean (June 2026), the other characters 99.4–99.8% (July 2026) · whole runs 1,232/1,250 clean (July 2026) |
| Throughput | one core: ~59k random-play fights/s · ~1.3M engine actions/s · whole machine (24 cores / 48 threads): ~1.4M fights/s · ~31M actions/s · vs the shipped game run headless, both kept busy: ~1,500× per core, ~2,430× whole machine |
| MCTS | the browser hint's search: 1,000 sims ≈ 14 ms native, one thread · ≈ 18 ms as WASM in Chromium on the same machine · the advisor's search: 8,000 sims ≈ 275 ms on one thread, ≈ 40 ms on eight, on an Act-1 boss |
| Training | a typical PPO iteration (Ironclad, 960 games, 46 workers, one RTX 3090) ≈ 208 s, 2.9× the previous engine · 150 iterations ≈ 9 h |
| Code | ~360k lines of Rust · ~116k Python · ~56k TypeScript · 7,000+ Rust tests, 3,000+ Python, 1,400+ web |

Performance: one AMD EPYC 7443P server (24 cores / 48 threads), 14 October
2026, plain release builds (the training row uses the profile-guided build
training runs on); one core = one pinned process with its second hardware
thread idle. The throughput row is random play: Ironclad's starter deck and
Burning Blood against each of the 22 Act-1 Overgrowth encounters. Earlier
versions of this README said ~23,000× the headless game: that timed one
cold game process per fight, mostly start-up and the harness's polling
sleeps. Full results:
[benchmarks page](https://tfriedel.github.io/spire-mentor/benchmarks.html);
method and commands:
[`sts2-rs/docs/perf/2026-10-14-benchmarks.md`](sts2-rs/docs/perf/2026-10-14-benchmarks.md).

**Run-AI champions** (simulator, MCTS combat at the real-game advisor's
budget of 8,000 sims per decision, Act 1 rolled, one fresh seed panel for all five,
measured October 2026; 200 runs per cell at A0–A8, 500 at A9–A10):

| Character | A0 | A4 | A8 | A9 | A10 | Mean A0–A10 | Promotion gate (vs previous champion) |
|---|---|---|---|---|---|---|---|
| Defect | 93.0% | 90.5% | 70.0% | 39.2% | 22.0% | **76.1%** | +5.5 pp, p = 2e-6 |
| Ironclad | 89.5% | 86.5% | 70.0% | 39.0% | 11.0% | **74.9%** | +5.2 pp, p = 1e-6 |
| Silent | 78.5% | 67.5% | 55.5% | 20.0% | 9.0% | **59.9%** | +5.6 pp, p = 7e-7, all 11 levels up |
| Regent | 67.5% | 66.0% | 43.5% | 15.6% | 6.2% | **53.8%** | +4.6 pp, p = 1e-4 |
| Necrobinder | 66.5% | 62.5% | 44.5% | 13.8% | 5.6% | **51.7%** | +6.8 pp, p = 6e-10, all 11 levels up |

Every level is in the
[docs](https://tfriedel.github.io/spire-mentor/#ai).

**In the real game** (the mod's advisor driving `SlayTheSpire2.exe`, 40 runs
each, random seeds, no cheats, 8,000 sims per combat decision, August 2026
champions): Ironclad A8 **32.5%**, Silent A8 **27.5%**.

## The browser workbench

![The live regret coach flagging a move as it is made](docs/site/img/wb-regret-coach.webp)

A React front end over the Rust engine compiled to WebAssembly, running in a
Web Worker; saves live in IndexedDB, and it installs as an offline PWA.

- **Play** any character, seed and ascension, with the real card text and —
  after `just import-game` — the real art and sound.
- **Rewind and branch** any decision; the run tree keeps every fork.
- **Ask the AI** for a combat hint, open the MCTS search tree, rank the
  choices on a card reward, shop or event, or let it play the best line.
- **Coach**: the live regret coach flags a weak move the moment you make it.
- **Autobattle**: you make the run-level decisions (paths, cards, shops,
  events) and the AI fights every combat; each stop shows what changed
  since the last one.
- **Watch the PPO champion** play a run to the end; take over at any point.
- **Review** a finished run for its biggest blunders: regret per decision
  and a win-probability chart.
- **SimLab**: build any single fight and see what the search thinks of
  every move, or replay it many times for the spread of outcomes and
  seed-paired strategy studies.
- **Lab tools**: replay viewer, MCTS-runs browser, deck analysis, and
  *state import* — paste a position copied from the live game and analyse
  it here.

### Workbench quickstart

From a fresh clone you need `rustup`, `just` 1.38+, `uv`, Node 20+ and a
`wasm-bindgen` CLI matching the crate
(`cargo install wasm-bindgen-cli --version 0.2.120 --locked`); the pinned
toolchain brings Rust 1.96 and the `wasm32-unknown-unknown` target, and the
first build runs `npm install`. On Windows, `.\install-windows-deps.ps1`
installs all of it except `rustup`, `uv`, git and VS Build Tools (see
[Windows](#windows)).

```sh
just import-game          # optional: real art and sound from your copy of the game
just workbench-local      # build + serve on http://127.0.0.1:4173
```

Skip `import-game` and the workbench still runs with the real card text and
placeholder art. With it, add `ilspycmd`, `cwebp` and `ffmpeg` — details in
[Quickstart](#quickstart) and
[docs/bring-your-own-game.md](docs/bring-your-own-game.md).

## Spire Mentor — the AI inside the real game

![Route advice from the run policy on the real game's map](docs/site/img/mod-hud-map.webp)

A C# mod for Slay the Spire 2 ([`spire-mentor-mod/`](spire-mentor-mod/)) that connects
the running game to the AI stack over a local socket. An idle mod leaves the
game vanilla.

- **Advisor mode** — you play; the mod shows ranked moves with MCTS
  statistics, card-reward and shop scores, the recommended map route and
  an estimated win chance.
- **AI turns** — `F6` lets the AI take one decision, `F7` lets it play until
  you stop it. Autopilot plays whole runs unattended.
- **Rewind** — every room entry is snapshotted with its RNG state, so you can
  undo a combat turn (`Ctrl+Z`), restart a room, or go back to any earlier
  room — bit-exact, including fights already finished.
- **Workbench link** — open the current game state in the workbench, browse
  the live game's tree and rewind the real game from the page.
- A toolbar, an in-game help panel (`F1`), an action log, speed controls and
  a settings page.

Build, install and protocol: [`spire-mentor-mod/README.md`](spire-mentor-mod/README.md),
[`spire-mentor-mod/PROTOCOL.md`](spire-mentor-mod/PROTOCOL.md).

## How it fits together

| Subproject | Language | What it is |
|---|---|---|
| [`sts2-rs/`](sts2-rs/) | Rust | The engine, MCTS, run-AI with ONNX policy inference, CLI, PyO3 bindings, and the workbench's engine layer (WASM). Builds and tests with nothing else present. |
| [`sts2-pycli/`](sts2-pycli/) | Python + C# + TS | The integration hub: the headless C# harness around the game assembly (your copy) that serves as the parity oracle, the parity suite, PPO training, the workbench web app, the real-game advisor, and an MCP server so an LLM can play. |
| [`spire-mentor-mod/`](spire-mentor-mod/) | C# | The Spire Mentor mod. |
| [`sts2-wiki/`](sts2-wiki/) | Python / Astro | Extraction tooling: decompiles your game copy, extracts data, art and sound, and can generate a static wiki. |
| `scripts/import_game.py` | Python | `just import-game` — produces every game-derived input from your install. |

Architecture: [ARCHITECTURE.md](ARCHITECTURE.md). Full documentation with
screenshots: **[tfriedel.github.io/spire-mentor](https://tfriedel.github.io/spire-mentor/)**
(source in [`docs/site/`](docs/site/)).

## What's in the box, and what isn't

> Slay the Spire 2 is © Mega Crit; this project is not affiliated with or
> endorsed by them. The repository carries the game's **text and extracted
> data**, so the engine, the AI and the workbench work out of the box, but
> **no art, audio, game assemblies or decompiled code**. The tooling extracts
> those from **a copy of the game you own**
> ([docs/bring-your-own-game.md](docs/bring-your-own-game.md)) into
> git-ignored paths. See [NOTICE](NOTICE) and [PROVENANCE.md](PROVENANCE.md).

## Quickstart

Three tiers; each needs more than the last. Prerequisites and details:
[docs/bring-your-own-game.md](docs/bring-your-own-game.md).

**Engine only — no game needed.** Rust 1.96 (pinned by
`sts2-rs/rust-toolchain.toml`), `cargo-nextest`, `just` (1.38+). `sccache` is
optional — `export RUSTC_WRAPPER=sccache` to use it.

```sh
just rs-check      # cargo check --workspace --all-targets
just rs-test       # cargo nextest run --workspace  (never bare `cargo test`)
just rs-doc-test   # cargo test --workspace --doc   (nextest cannot run doctests)
```

**Oracle — the C# parity harness.** Add the game, the .NET 9 SDK, `uv`.

```sh
just import-game /path/to/Slay\ the\ Spire\ 2 --tier oracle
cd sts2-pycli && uv sync --dev && just build-cs && uv run pytest tests
```

**Workbench with real art and sound (and the decompiled source).** Add
`ilspycmd` 9.1 (`dotnet tool install -g ilspycmd --version 9.1.0.7988`),
Node 20+, a `wasm-bindgen` CLI matching the crate
(`cargo install wasm-bindgen-cli --version 0.2.120 --locked`), `cwebp`
(libwebp) and `ffmpeg`. On Windows the sound needs nothing else; on Linux it
also needs the FMOD SDK (`FMOD_SDK_ROOT`, see
[docs/bring-your-own-game.md](docs/bring-your-own-game.md)).

```sh
just import-game          # finds the game through Steam; art, data and sound
just workbench-local      # build + serve on http://127.0.0.1:4173
```

`workbench-local` builds `sts2-pycli/web/dist` (the engine runs in the
browser as wasm — no backend) and serves it on this machine only.
`just workbench-server` is the Python-backed alternative on :8077.

**Please run the workbench locally and don't host it publicly.** With real
art the build contains the game's images and sound, extracted from your copy
for your own use. The servers bind to `127.0.0.1` by default; `STS2_HOST`
(Python servers) and `STS2_WEB_HOST` (Vite) open them to your LAN.

Without the game the workbench still builds and runs with the real card
text; every image is a neutral placeholder tile and there is no sound until
`just import-game` extracts the art and the audio (on Windows with the FMOD
runtime the game ships; on Linux with the FMOD SDK — see
[docs/bring-your-own-game.md](docs/bring-your-own-game.md)).

**The mod.** Windows, the game at v0.107.1, the .NET 9 SDK and
[BaseLib](https://github.com/Alchyr/BaseLib-StS2) v3.4.1 in
`<game>/mods/BaseLib/` (on Windows, `install-windows-deps.ps1` installs both):
`cd spire-mentor-mod && dotnet build` builds it and
installs it into the game's `mods/` folder; the in-game advisor also needs
`cd sts2-pycli && uv sync` once. Details in
[`spire-mentor-mod/README.md`](spire-mentor-mod/README.md).

## Windows

Everything above runs natively from **PowerShell** — the root `justfile` runs
its recipes under PowerShell on Windows. From a fresh clone:

```powershell
.\install-windows-deps.ps1                  # the toolchain (winget / cargo / dotnet), idempotent
# new terminal, so the tools are on PATH
just import-game                            # finds the game through Steam
just workbench-local                        # http://127.0.0.1:4173
cd spire-mentor-mod; dotnet build           # the mod (BaseLib first: spire-mentor-mod\README.md)
```

The script installs everything except `rustup`, `uv`, git and VS Build Tools
(the Rust linker), which it checks for —
[BUILDING-WINDOWS.md](BUILDING-WINDOWS.md) has the details. The subproject
justfiles (`sts2-rs/`, `sts2-pycli/`, `sts2-wiki/`) are bash: run those from
Git Bash.

## Testing

- Rust: `cargo nextest run --workspace` **and** `cargo test --workspace --doc`
  from `sts2-rs/`. Bare `cargo test` shares one process across tests and
  produces order-dependent false failures here. Game-dependent tests skip
  when the game data is absent.
- Python: `uv run pytest tests` from `sts2-pycli/`. Tests that need the C#
  oracle carry the `oracle` marker and skip without `lib/sts2.dll`;
  `-m "not oracle"` is the game-free subset CI runs.
- Web: `npm test` from `sts2-pycli/web/` (vitest); suites that need the
  extracted audio/art skip without them.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) — including how this repository relates
to the private development repo it is synced from.

## Credits

- `sts2-pycli` began as a fork of [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli)
  by Hao Wu (MIT).
- Spire Mentor started as a port of zhiyue's STS2BridgeMod from
  [zhiyue/sts2-rl-agent](https://github.com/zhiyue/sts2-rl-agent).
- The mod's settings page uses Alchyr's
  [BaseLib](https://github.com/Alchyr/BaseLib-StS2).

## License

MIT — [LICENSE](LICENSE); `sts2-rs/`, `sts2-pycli/`, `spire-mentor-mod/` and
`sts2-wiki/` carry their own MIT files. The grant covers the original work
only; nothing from the game is licensed or included.
