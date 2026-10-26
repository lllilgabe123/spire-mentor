# Open-source plan — decisions and the export flow

The public repository is **`github.com/tfriedel/spire-mentor`**; this private
monorepo is `tfriedel/spire-mentor-private`. The public repo is a
**fresh-history snapshot** of this tree with every game-derived file left
out, produced by `release/export_unit.py` from the single manifest
`release/units/unit-mono.toml`, and refreshed by squashed sync commits.

Companion documents:

- [PROVENANCE.md](../PROVENANCE.md) — the authoritative "what is and is not
  redistributable" inventory.
- `release/README.md` — the exporter: gates, transform, overlay, sync.
- [docs/bring-your-own-game.md](bring-your-own-game.md) — how a user of the
  public repo regenerates the game-derived inputs from their own copy.

## Decisions (settled 2026-09-13)

The eight owner decisions the first draft of this plan left open, as
resolved. They supersede the earlier four-unit (A–D) design, whose manifests
were deleted (recoverable with
`git show 44f1adcaa:release/units/unit-b-bridge.toml` etc.).

| # | Decision | Resolution |
|---|---|---|
| D0 | Game text | **Ships.** Localization and the extracted data JSON are part of the public repo (owner decision 2026-09-13) — the engine, AI and workbench must work without the game. Art, audio, assemblies and decompiled code do not ship. |
| D1 | Decompiled C# quoted in original code/docs | **Strip at export.** Quotes sit in code fences tagged `csharp` by convention; the exporter replaces each block with a placeholder line and fails on a `text`-tagged fence that still looks like C#. Path citations and namespace names stay as references, ratcheted. The engine still re-expresses the game's rules; contacting Mega Crit for a non-objection is a separate, open question. |
| D2 | History export strategy | **Fresh-history snapshot + squashed syncs.** The private history is irrecoverably mixed (deleted game paths under two prefix sets, ~4 GB of `.beads/issues.jsonl` with pasted C#); it is never filtered or published. Each sync = export → mirror onto the public checkout → one `sync: spire-mentor-private@<sha>` commit. |
| D3 | Champions manifest ownership | **Moot** — one public monorepo keeps `champions.toml` where it is; the exporter reads it to select exactly the deployed `.onnx` files (sha256-verified). |
| D4 | Publishing a `sts2_rs` wheel | **Moot** — the public repo is the whole monorepo; `uv sync` builds the wheel from the sibling path dep. |
| D5 | spire-mentor-mod donor license | **Resolved 2026-10-11 — ships.** Game side: the files that mirrored the game's AutoSlay framework (`SimAutoSlayer.cs` and four handlers) were rewritten clean-room from a behavioural spec, and the exporter's decompiled-overlap gate (`[gates] decomp_*` in the manifest) fails any exported `.cs` file that reproduces decompiled code. Donor side: zhiyue/sts2-rl-agent has no LICENSE file (its README only says the project is "for research and educational purposes"), so every donor-derived region was also rewritten clean-room (round 2, 2026-10-11); the same gate compares every exported `.cs` file against the pinned donor source (`donor_dirs`, fetched by `just fetch-donors`). `spire-mentor-mod/LICENSE` is plain MIT with a credit line. Was: excluded until a donor grant or a clean-room rewrite existed. |
| D6 | Scraped run corpus and derived nets | **Excluded** (corpus, codex fixtures, `value_human_a0*`, `*ranker*`, synergy catalogs). Exceptions: the MCTS playout nets `sts2-rs/data/rollout_nets/ironclad.json` (owner decision 2026-10-18), `defect.json` (2026-10-21, GH #825), `regent.json` (2026-10-21, GH #822), `necrobinder.json` and `silent.json` (2026-10-21, GH #823, #824) ship (trained in the simulator, the runs gave only fight start positions); and the Ironclad MCTS move prior `sts2-rs/data/prior_nets/ironclad.json` ships (owner decision 2026-10-25, GH #830). Unlike the playout nets it was trained by behaviour cloning on human actions from community-uploaded replays; it ships as weights only. PPO champions (own-sim) ship as plain blobs; `.pt` checkpoints and retired nets do not. |
| D7 | Workbench Docker image | **Private** (`Dockerfile` excluded); the browser build is what the public repo produces. |
| D8 | CI | **Public repo only.** `release/public-overlay/.github/workflows/ci.yml` (and `pages.yml`, which publishes `docs/site/` to GitHub Pages) is added at export; the private repo keeps its no-Actions policy (ADR-0002). |

Also settled: one public monorepo (not four unit repos); no git-LFS in the
public repo (nothing that would need it ships); nets ship as plain blobs
under the 50 MB gate.

## The boundary, in one paragraph

Ships: `sts2-rs/` (the whole cargo workspace, own data), `sts2-pycli/`
(python, C# harness sources, tests, scripts, web app, the five champion
`.onnx`), `sts2-wiki/` tooling (scripts, tools, site templates), root docs,
`scripts/import_game.py`, **and the game's text and extracted data**
(localization, `sts2-wiki/data/v0.107.1`, the workbench bundle and goldens —
PROVENANCE.md §0) so the simulator, the AI and the workbench run out of the
box. Never ships: decompiled code, art, audio, the game assembly, the raw pck
extraction, the corpus-derived nets/catalogs, private infra (PROVENANCE.md
§1–§2). In the public repo those paths are git-ignored, and `just
import-game` regenerates them from the user's own game copy; the workbench
renders placeholder tiles until the art is extracted.

## Sync workflow

From the private checkout, on a green `main`:

```sh
just check-public                    # the gate (also part of the local gates)
just sync-public ../spire-mentor-public  # export -> mirror onto the public clone
cd ../spire-mentor-public && git status  # review
git add -A && git commit -m "sync: spire-mentor-private@<sha>" && git push
```

Outside contributions land in the public repo as normal PRs, are re-applied
to the private `main` (`git format-patch` / cherry-pick), and reappear in
the next sync commit — the public tree never diverges for long because the
mirror is unconditional. Because it is unconditional, the sync guards against
a forgotten port: any path changed on the public side since the newest sync
commit must come out of the export byte-identical to public HEAD, or the sync
refuses (listing the paths and public commits); it also refuses a dirty public
worktree or one with no sync commit. `--allow-revert` overrides with a warning.
Commit syncs with the exact `sync: spire-mentor-private@<sha>` subject — the guard
keys on it.

## Recorded non-actions

- **Editorial wiki notes** (`sts2-wiki/scripts/{card_notes,monster_notes,event_enrichments}.py`)
  stay private: they quote game content. The generators import them
  optionally.
- **Third-party OSS DLL attribution** — `sts2-pycli/lib/` never ships, so
  none is scheduled.
