# PROVENANCE — what is and is not redistributable

This monorepo mixes original work with content derived from the commercial
game **Slay the Spire 2** ((c) Mega Crit). This file is the single statement
of which is which. The public repository (`github.com/tfriedel/spire-mentor`) is
a snapshot export that contains only the original work; the export boundary
is enforced mechanically by `release/export_unit.py` and its manifest
`release/units/unit-mono.toml` — this document is the rationale behind them.
The decisions that shaped the boundary are recorded in
`docs/open-source-plan.md`.

## 0. The rule

**Game text and extracted data ship; everything else from the game does
not.** The public repository carries the game's English localization and the
JSON the extractors derive from the decompilation (card / relic / potion /
monster / event data, the workbench bundle, recorded goldens) so the
simulator, the AI and the workbench work out of the box. It carries no
decompiled code, no art, no audio, no game assembly and no raw pck
extraction: those are §1 below, git-ignored publicly, and `just import-game`
regenerates them from a copy of the game the user owns
(`docs/bring-your-own-game.md`). Verbatim decompiled C# quoted in comments is
stripped at export (§1, last part).

## 1. Game-derived content that is NOT text — never redistributed

Stays in the private monorepo for development, is excluded from the export,
and is git-ignored in the public repository.

### sts2-wiki (the main boundary — effectively the whole subproject's data)

| Path | What it is |
|---|---|
| `sts2-wiki/decompiled/` | Full C# decompilations of two game builds (v0.103.2, v0.107.1). The mechanics source of truth for development; verbatim game code. |
| `sts2-wiki/extracted/` | Raw pck extraction (localization, atlases, FMOD banks); git-ignored everywhere. |
| `sts2-wiki/site/public/images/` | ~2,469 extracted art files (card atlases, portraits, Spine character/monster rigs — git-LFS in the private repo). |
| `sts2-wiki/site/public/audio/` | 566 extracted sound effects (Opus/WebM), rendered from the game's FMOD banks. Music and ambience are deliberately not extracted. |
| `sts2-wiki/docs/version-diffs/` | Build-to-build diffs derived from comparing decompilations (code). |

Only the generator/tooling code (`scripts/`, `tools/`, site templates) is
original work — see `sts2-wiki/LICENSE`, whose grant is scoped accordingly.

### Game content outside sts2-wiki

| Path | What it is |
|---|---|
| `sts2-pycli/lib/*.dll` | The game's main assembly (IL-patched headless build) and the OSS DLLs it ships with, vendored via git-LFS in the private repo. `setup.sh` produces them from a local install. |
| `sts2-pycli/web/public/preview/`, `preview-rig/` | PNG/`.skel` Spine art extracted from the game package. |
| `sts2-pycli/web/public/catalog*.json`, `archetypes_*.json` | Synergy graphs computed from the scraped run corpus (§2, D6). |
| `sts2-pycli/data/parity-cache/`, `data/run-parity-cache/` | zstd bundles (git-LFS) of recorded C# oracle execution traces — produced by running the real game engine. |
| `sts2-pycli/deploy/asset_manifest.json` | The art-store manifest (`build_asset_manifest.py`, private deploy tree). |

### Game text and data that DO ship (§0)

| Path | What it is |
|---|---|
| `sts2-pycli/localization_eng/` | ~46 JSON files — the game's full English localization, vendored from the engine-pinned build's pck by `scripts/vendor_localization.py` (keys only an older build shipped are retained for content the engine still implements). |
| `sts2-wiki/data/v0.107.1/` | Extracted game data JSON for the pinned build (cards, relics, potions, monsters, events, VFX bindings, power-flash sites). Other builds' data stays private. |
| `sts2-wiki/overrides/`, `site/src/content/`, `site/src/pages/patches/`, `scripts/{card_notes,monster_notes,event_enrichments}.py` | Wiki content and editorial notes describing / quoting game text. |
| `sts2-pycli/web/public/card_text.json`, `sfx_map.json`, `python/workbench/data/card_db.json` | Card text, the game <-> sound-event map, the deck analyzer's card database. |
| `sts2-rs/crates/sts2-workbench/data/game/` | The browser workbench's embedded bundle (localization copies, trimmed wiki manifests, the art manifest). `build.rs` substitutes empty stubs when it is absent. |
| `sts2-rs/crates/sts2-workbench/tests/golden/`, `sts2-pycli/tests/workbench/golden/` | Recorded workbench goldens (rendered game text). |

### Decompiled code quoted in original files

Doc comments and docs in `sts2-rs/` and `sts2-pycli/` cite the decompilation
by path (`../sts2-wiki/decompiled/v0.107.1/…/Foo.cs:56-69`) and, where a
port needed it, quote the decompiled C# verbatim inside a code fence tagged `csharp`
Convention: **every verbatim C# quote is fenced as csharp** — the
exporter strips those blocks (leaving one placeholder line) and fails on a
` ```text ` block that still looks like C#. Path citations and namespace names
(`MegaCrit.Sts2.Core.…`) are references, not content, and remain; their count
is ratcheted by `release/content-marker-baseline.json`.

**`spire-mentor-mod/` (D5, resolved 2026-10-11 — see the plan).** It started as a port of
zhiyue's STS2BridgeMod; the run driver and four room/screen handlers, which
had mirrored the game's own AutoSlay framework, were rewritten clean-room
from a behavioural spec. The exporter's decompiled-overlap gate fails any
exported `.cs` file that shares more than a handful of lines (single API
calls, interface signatures) with the game's decompilation. The mod links
the game's assemblies at build time (Harmony); none of them ships.
Every region derived from zhiyue/sts2-rl-agent (which has no license) was
rewritten clean-room as well; the exporter also compares every exported `.cs`
file against the pinned donor source (`just fetch-donors`), and
`spire-mentor-mod/LICENSE` is plain MIT with a credit line.

## 2. Excluded for other reasons

- **Scraped player runs and the nets trained on them (D6)** —
  `sts2-rs/data/run-corpus/files.tar.zst` (6,451 community-uploaded `.run`
  files, git-LFS), `sts2-pycli/python/synergy/tests/fixtures/codex/**`,
  `sts2-rs/data/runai/nets/value_human_a0*` and `nets/*ranker*` (trained on
  the corpus's outcome labels), and the synergy catalogs above.
  Redistribution rights for player-uploaded data are unresolved.
  Two exceptions ship. The first is the MCTS playout nets
  `sts2-rs/data/rollout_nets/ironclad.json` (owner decision, 2026-10-18),
  `sts2-rs/data/rollout_nets/defect.json` (owner decision, 2026-10-21, GH
  #825), `regent.json` (owner decision, 2026-10-21, GH #822), and
  `necrobinder.json` and `silent.json` in the same folder (the #825 decision
  extended by the owner, 2026-10-21, GH #823, #824). They were trained by the simulator's own search and playouts; the
  scraped runs supplied only the fight start positions (deck, relics, HP).
  The files hold network weights and no run data.
  The second is the Ironclad MCTS move prior
  `sts2-rs/data/prior_nets/ironclad.json` (owner decision, 2026-10-25, GH
  #830): the PUCT prior the Ironclad run agent uses in elite and boss fights
  (`sts2-rs/data/prior_nets/README.md`). Unlike the playout nets, it was
  trained on human play: by behaviour cloning on 137,357 human combat
  decisions, reconstructed from community-uploaded per-action replays. The
  rights for that underlying player data stay unresolved; the owner decided
  to ship the trained weights regardless. The file holds network weights and
  a card/potion action-key vocabulary, no run data (its training metadata
  was stripped). It is not one of the own-output trained artifacts of §3.
- **Private infrastructure** — `sts2-pycli/deploy/` (Terraform, GCP identity,
  the deploy justfile), `sts2-pycli/web/wrangler.toml`, the root `Dockerfile`,
  `.gcloudignore`, `.dockerignore`, `sts2-pycli/packaging/`.
- **Lab notebook and agent infrastructure** — `docs/experiments/`,
  `docs/plans/`, `docs/superpowers/`, `logs/`, every `experiments/` tree,
  `.beads/`, `.claude/`, `CLAUDE.md` / `AGENTS.md` / `CONTEXT.md` files,
  `release/` itself, training checkpoints (`*.pt`), retired champion nets.

### Third-party OSS DLLs in `sts2-pycli/lib/`

`0Harmony.dll`, `MonoMod.*.dll`, `Sentry.dll`, `SmartFormat*.dll`,
`Steamworks.NET.dll`, `System.IO.Hashing.dll` are OSS-licensed vendored
binaries. They sit in the never-ship `lib/` directory (the user's `setup.sh`
copies them from the game install), so no attribution files are scheduled.

## 3. Original work — redistributable

- **`sts2-rs/`** — the Rust combat/run engine, MCTS, run-AI, CLI, PyO3
  bindings, the browser-workbench port, benchmarks. MIT (`sts2-rs/LICENSE`).
- **`sts2-pycli/`** — the Python parity driver, CLI, PPO training stack,
  workbench, headless C# runner sources (`src/`, including the clean-room
  `GodotStubs`), the browser build. MIT (`sts2-pycli/LICENSE`). The grant
  covers the code, not the game content co-located in the paths above.
- **`sts2-wiki/`** generator/tooling — `scripts/`, `tools/`, site templates.
  MIT (`sts2-wiki/LICENSE`). Users bring their own game copy for
  `just decompile` / `just extract`.
- **Trained artifacts** (own outputs):
  - PPO champions: the `.onnx` nets named by
    `sts2-pycli/data/ppo/champions.toml`, trained on the project's own Rust
    simulator; each carries a `*.provenance.md` sidecar. The exporter selects
    exactly those files and verifies the recorded sha256.
  - `sts2-rs/data/runai/nets/` value nets other than the corpus-trained ones,
    `sts2-rs/data/value_policy_net/`, the combat position books and dominance
    pairs, own-sim seed panels under `sts2-rs/data/seed-panels/`.
- **Kreon font** (`sts2-pycli/web/public/fonts/Kreon.ttf`) — third-party,
  SIL OFL 1.1, redistributable with the license text vendored alongside it
  (`fonts/OFL.txt`).
- **FastNoiseLite** (`sts2-wiki/scripts/noise_oracle/FastNoiseLite.h`) —
  third-party, MIT, header retained with its copyright line.

## 4. How the boundary is enforced

`python3 release/export_unit.py --check --unit mono` (root `just check-public`)
validates the working tree against the manifest: allowlist + a hard-coded
deny list of every path in §1–§2, no LFS pointers or `filter=lfs` attributes,
the csharp-fence transform and its gate, an identifier gate (home paths,
personal accounts, live infrastructure names), a 50 MB size gate, and the
content-marker ratchet. The same checks run again on the exported tree.
`just sync-public <checkout>` mirrors a fresh export onto the public clone for
a squashed sync commit. Details: `release/README.md`.
