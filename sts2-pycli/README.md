# sts2-pycli

<details open>
<summary><b>English</b></summary>

A CLI for Slay the Spire 2, plus the Python AI stack built around it.

Runs the real game engine headless in your terminal — all damage, card effects, enemy AI, relics, and RNG are identical to the actual game. Everything is unlocked from the start: all characters, cards, relics, potions, and ascension levels — no timeline progression required.

This project began as a fork of [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli) by Hao Wu (MIT — see [LICENSE](LICENSE)) and has since grown into the Python half of the [spire-mentor](https://github.com/tfriedel/spire-mentor) monorepo, sitting next to its siblings `sts2-rs` (Rust engine), `sts2-wiki` and `spire-mentor-mod`.

![demo](docs/demo_en.gif)

## What lives here (concern map)

| Concern | Where | One-liner |
|---|---|---|
| C# headless oracle | `src/Sts2Headless/`, `src/GodotStubs/` | Runs the real, IL-patched game DLLs from `lib/` headless over a JSON stdin/stdout protocol — the authoritative reference for game mechanics. |
| Parity harness | `python/parity_audit.py`, `python/parity_fuzz*.py`, `python/run_parity.py`, … | Locksteps the sibling Rust engine (`../sts2-rs`) against the C# oracle and buckets divergences. Map: `docs/parity-suite.md`. |
| PPO training | `python/ppo/`, `data/ppo/champions.toml` | Full-run self-play PPO for the run policy; champion nets are promoted via `just promote-champion`. Playbook: `../docs/ppo-champion-training-playbook.md`. |
| Web workbench | `python/workbench/` + `web/` | FastAPI backend + Vite frontend for replaying recorded runs, regret review, and live advice. |
| Real-game bridge client | `python/realgame_*.py`, `python/sts2_realgame.py` | TCP clients for the Spire Mentor mod (`../spire-mentor-mod/PROTOCOL.md`) that drive or observe the live game process. |
| Advisor exe | `packaging/SimAdvisor.spec`, `scripts/build_advisor_exe.ps1` | PyInstaller-frozen `realgame_advisor` shipped alongside the Spire Mentor mod. |
| MCP server | `python/sts2_mcp.py` | Lets Claude (or any MCP client) play the game through typed tools. |

`python/` itself is a flat script tree with several packages inside; see [`python/README.md`](python/README.md) for the per-concern file map and entry points.

## Setup

Requirements:
- [Slay the Spire 2](https://store.steampowered.com/app/2868840/Slay_the_Spire_2/) on Steam
- [.NET 9+ SDK](https://dotnet.microsoft.com/download)
- Python 3.13+

```bash
git clone git@github.com:tfriedel/spire-mentor.git
cd spire-mentor/sts2-pycli
./setup.sh      # copies DLLs from Steam → IL patches → builds
```

Or just run `python3 python/play.py` — it auto-detects and sets up on first run.

Cloning the whole monorepo matters: `pyproject.toml` declares the `sts2_rs`
hint/parity extension as an editable path dependency on the sibling
`../sts2-rs/crates/sts2-py`, so `uv sync` expects the siblings to be present.

### Game-content provenance

`lib/` (the game DLLs, IL-patched) and `localization_eng/` (English
localization data) are **content extracted from Slay the Spire 2**. They are
not covered by this project's license and must never be redistributed — they
exist in a checkout only so the headless oracle can run, and the supported way
to obtain them is `./setup.sh` against **your own game install** (it copies and
IL-patches the DLLs from your Steam copy). See
[`../PROVENANCE.md`](../PROVENANCE.md) for the repo-wide statement of what is
and is not redistributable.

## Play

```bash
python3 python/play.py                        # interactive (English)
python3 python/play.py --ascension 10         # Ascension 10
python3 python/play.py --character Silent      # play as Silent
```

Type `help` in-game:

```
  help     — show help
  map      — show map
  deck     — show deck
  potions  — show potions
  relics   — show relics
  quit     — quit

  Map:     enter path number (0, 1, 2)
  Combat:  card index / e (end turn) / p0 (use potion)
  Reward:  card index / s (skip)
  Rest:    option index
  Event:   option index / leave
  Shop:    c0 (card) / r0 (relic) / p0 (potion) / rm (remove) / leave
```

## JSON Protocol

For programmatic control (AI agents, RL, etc.), communicate via stdin/stdout JSON:

```bash
dotnet run --project src/Sts2Headless/Sts2Headless.csproj
```

```json
{"cmd": "start_run", "character": "Ironclad", "seed": "test", "ascension": 0}
{"cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 0}}
{"cmd": "action", "action": "end_turn"}
{"cmd": "action", "action": "select_map_node", "args": {"col": 3, "row": 1}}
{"cmd": "action", "action": "skip_card_reward"}
{"cmd": "quit"}
```

Each command returns a JSON decision point (`map_select` / `combat_play` / `card_reward` / `rest_site` / `event_choice` / `shop` / `game_over`). All names are in English.

## MCTS hint integration (optional)

`sts2-pycli` can run [`sts2-rs`](../sts2-rs/) as an in-process MCTS hint sidecar. The C# `Sts2Headless` engine remains authoritative for game state; `sts2-rs` only scores the current decision point and surfaces ranked suggestions next to each numbered choice. Disable entirely with `--no-hints`.

### Build

`./setup.sh` (the same script you run for `.NET` setup) also builds and installs the `sts2_rs` PyO3 extension into the active venv via [maturin](https://github.com/PyO3/maturin). Set `STS2_SKIP_RUST=1` to skip just the Rust step if your environment lacks a Rust toolchain — combat continues to work, only the hint sidecar is disabled.

Prerequisites for the Rust step:
- Rust 1.99+ (`rustup install stable`; `../sts2-rs/rust-toolchain.toml` pins the exact toolchain)
- `cargo install maturin`
- An activated venv (`uv sync --dev`, then `source .venv/bin/activate`) — `setup.sh` refuses to install into an unset or unrelated venv.

### Flags

`play.py` accepts the following MCTS hint flags (defaults reflect the plan at `../sts2-rs/docs/plans/2026-05-14-sts2-cli-mcts-hint-integration.md` §"CLI surface"):

| Flag | Purpose |
|---|---|
| `--mcts-sims N` | MCTS simulations per legal action (knee 300–3000). |
| `--mcts-depth D` | Max rollout depth. |
| `--mcts-c-puct F` | PUCT exploration constant. |
| `--mcts-seed N` | MCTS-side RNG seed (0 = derive from combat seed). |
| `--mcts-threads N` | Parallel rollout threads. |
| `--mcts-fair` | Use the fair-info rollout policy (no draw-pile cheating). |
| `--mcts-cheat-draw-order` | Allow rollouts to see the upcoming draw order. |
| `--no-hints` | Disable the hint sidecar entirely for the session. |
| `--hint-on-demand` | Suppress inline tags; expose `/hint` REPL verb instead. |
| `--hint-timeout-ms N` | Per-decision wall-clock cap (falls back to no hint on overrun). |

> **NOTE:** The flag set is finalised in bead `sts2-rs-vjd1`. Verify the live flag names with `python3 python/play.py --help` if anything below disagrees.

### MCP tool

The MCP server adds a new tool **`recommend_combat_mcts`** (sidecar of the existing heuristic `recommend_combat` tool, which is untouched):

```
recommend_combat_mcts(sims=300, depth=10, c_puct=1.4, seed=0, threads=1, fair=False)
```

Returns a list of `(action_index, win_prob, sims_used, pv_summary)` tuples ranked by `win_prob`. On `UnsupportedStateError` (e.g. an unknown card id or a wire-schema mismatch — see [`../sts2-rs/docs/schema-version-policy.md`](../sts2-rs/docs/schema-version-policy.md)), the tool returns:

```json
{"error": "<message>", "fallback": <recommend_combat heuristic result>}
```

so callers always get *some* answer.

### Caveat — hints rank, they don't predict

The sidecar runs its own RNG cursor (seeded from `state.seed + turn`); the C# engine's RNG is independent. Rollouts therefore do **not** reproduce the real game's outcome distribution. Win-prob percentages are useful for **ranking** the legal actions; they are **not** absolute win-rate forecasts. See [`../sts2-rs/tests/parity/known_divergences.md`](../sts2-rs/tests/parity/known_divergences.md) §"2026-05-14 mcts-hint-sidecar-rng-divergence" and the design spec at [`../sts2-rs/docs/plans/2026-05-14-sts2-cli-mcts-hint-integration.md`](../sts2-rs/docs/plans/2026-05-14-sts2-cli-mcts-hint-integration.md).

### Troubleshooting

- **Stale wheel after a Rust-side bump.** `setup.sh` uninstalls and reinstalls the `sts2_rs` wheel each run, but if you bypassed setup.sh, force-clean it:
  ```bash
  pip uninstall sts2_rs -y && ./setup.sh
  # or with uv:
  uv pip uninstall sts2_rs && ./setup.sh
  ```
- **Missing maturin.** `cargo install maturin` (or `pipx install maturin`).
- **Wrong venv.** `setup.sh` checks `$VIRTUAL_ENV` and fails loud if it doesn't look like sts2-pycli's. Activate the right one (`source .venv/bin/activate`) and retry.
- **Schema mismatch.** If sts2-pycli emits an older `wire_version` than the installed `sts2_rs` expects, the integration logs the gap and disables hints for the session — combat continues unaffected. Rebuild via `./setup.sh` or pin a compatible `sts2-rs` commit.

## Game Logs

Every run is automatically logged to `logs/` as a JSONL file (one JSON per line), recording each game state and action with timestamps. Logs older than 7 days are cleaned up automatically.

```bash
python3 python/play.py --no-log    # disable logging
```

**When filing a bug report, please attach the relevant log file from `logs/`** — it contains the full step-by-step game state needed to reproduce the issue.

## Development

Python tooling is managed with [uv](https://docs.astral.sh/uv/). The included `justfile` wraps the common commands (requires [`just`](https://github.com/casey/just)):

```bash
just setup            # uv sync + install pre-commit hooks (prek)
just lint             # ruff check
just format           # ruff format
just typecheck        # ty (Astral) — scoped to new code
just deptry           # check for unused/missing deps
just test             # pytest (requires the C# runner to be built)
just build-cs         # dotnet build of src/Sts2Headless
just regression-test  # 5 games × 5 characters (slow)
just --list           # all recipes
```

Pre-commit hooks (`sync-with-uv`, `uv-lock`, ruff, gitleaks, ty) run automatically on `git commit` via [prek](https://github.com/j178/prek), a Rust drop-in for `pre-commit`.

There is **no CI in this monorepo** — nothing runs on pull requests, by
deliberate decision (see the root `CLAUDE.md` for the rationale). The local
gates are the only gates: `just check` (or its alias `just ci`) runs
format-check + lint + typecheck + deptry + pytest. A GitHub Actions workflow
for a standalone export of this subproject is preserved, inert, at
`docs/ci-templates/ci.yml`; it only activates if deliberately restored to
`.github/workflows/` in such an export.

## Workbench (GUI)

A browser dashboard for visualizing AI play over a recorded run (replay
core; see `docs/superpowers/specs/2026-05-24-sts2-gui-workbench-design.md`).

```bash
just workbench       # backend (FastAPI :8077); serves steps + game art
just workbench-web   # frontend (Vite :5173)
# open http://localhost:5173, pick a run, scrub combat steps, click "recommend"
```

Art is served from `../sts2-wiki/site/public/images` (override with
`STS2_WIKI_IMAGES`). Logs come from `logs/` (override with `STS2_LOGS_DIR`).

### Post-combat regret review (the learning engine)

Load a recorded run and click **📉 regret review** (or `POST /api/regret-review`)
to replay every combat decision, diff it against the MCTS line, and surface the
1–2 plays that cost the most win-probability —
e.g. *"Turn 3: Strike instead of Defend dropped P(win) 71%→58% (−13 pts)"*.
It reads each step's recorded MCTS stats when present, and re-runs the search
(faithful fixture replay when available) otherwise — pass `{"recompute": true}`
to force a fresh search. In a live run, `POST /api/live/last-move-regret` flags a
blunder right after you make one, comparing the move you just played against the
recommended line. Win-probabilities are the same `wins/visits` the hint panel
shows.

### Inspecting MCTS runs

To record what the engine's MCTS does over a combat and inspect *why it won or
lost* — as agent-readable Markdown, a batch loss summary, or the workbench's
**MCTS Runs** page — see **[docs/workbench-mcts-runs.md](docs/workbench-mcts-runs.md)**.

```bash
# record MCTS over a .run-source fixture, then read it
WHEEL_VENV_NO_BUILD=1 scripts/wheel-venv.sh run -- python -m workbench.record --fixture <fixture.toml> --out logs
just aggregate logs            # which runs we lost + loss patterns
just report logs/<run>.jsonl   # one run, per-decision MCTS table (Q / W-L-T / PV)
```

This replaces the deprecated Rust `sts2-fight-replay` HTML flow.

## SimLab

Two tools on one page (`/simlab.html`, linked from the workbench), both built
directly on the Rust engine:

- **Sandbox** — a position editor in the spirit of a chess analysis board:
  character, ascension, turn, encounter and seed; the player's HP / block /
  energy / relics / potions / powers; every card by pile (draw pile in deal
  order); each enemy with HP, block, powers and its telegraphed move. The
  engine rebuilds the position on every edit ("Built." or exactly what it
  refused). **Play it out** sweeps the exact board — trials per strategy, one
  cell per legal first move if you like — and **search** runs the workbench's
  MCTS on it. The workbench's **🧪 SimLab from here** opens the sandbox on the
  live board, and a Spire Mentor **Copy state** paste loads straight into the
  Position JSON box. See `docs/live-state-import.md`.
- **Studies** — the batch sweep: deck variants × encounters × ascensions from
  turn 1 under five strategies — `random`, `block_first`, `attack_first`,
  `min_overblock` (minimize wasted Block) and `mcts` (engine MCTS) — seed-paired
  so strategies are compared on identical fights, with damage-taken
  distributions and paired per-seed deltas. Inspired by Caleb Gannon's
  quantitative STS2 strategy analysis.

```bash
just sim --name demo --encounter MAWLER_NORMAL --seeds 1000   # CLI sweep
just sim-bench                                                # perf bench
just sim-ui                                                   # webapp on http://localhost:8078/simlab.html
```

See `docs/superpowers/specs/2026-06-04-simlab-design.md` for the original
sweep design.

## Supported Characters

| Character | Status |
|---|---|
| Ironclad | Fully playable |
| Silent | Fully playable |
| Defect | Fully playable |
| Necrobinder | Fully playable |
| Regent | Fully playable |

## Architecture

```
Your code (Python / JS / LLM)
    │  JSON stdin/stdout
    ▼
src/Sts2Headless (C#)
    │  RunSimulator.cs
    ▼
sts2.dll (game engine, IL patched)
  + src/GodotStubs (replaces GodotSharp.dll)
  + Harmony patches (localization)
```

</details>
