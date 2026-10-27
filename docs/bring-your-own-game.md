# Bring your own game

This repository ships the game's **text and extracted data** (localization,
card/relic/monster JSON, the workbench bundle) but no game code, decompiled
code, art, audio or assemblies. The Rust engine, the AI stack and the
workbench build and run without the game — the workbench shows a neutral
placeholder tile wherever an image would go, and is silent. The rest is
extracted, on your machine, from the copy you own, into git-ignored paths.

You need the game for: the C# parity oracle (`oracle` tier), the decompiled
source the extractors and the mechanics work read plus real art
(`workbench` tier), sound (`sound` tier).

## What you get per tier

| Tier | Needs | Produces | Unlocks |
|---|---|---|---|
| `engine` | nothing | — | the engine, the AI, the workbench (placeholder art, no sound); every oracle-dependent test skips |
| `oracle` | the game, .NET 9 SDK, `uv`, `just` | `sts2-pycli/lib/<version>/` (IL-patched game DLLs); refreshes `localization_eng/` | the C# oracle (`just build-cs`), the parity harness, the full pytest suite |
| `workbench` | + `ilspycmd` | `sts2-wiki/decompiled/`, `sts2-wiki/site/public/images/`; refreshes `sts2-wiki/data/<ver>/`, `sfx_map.json`, card text/db, the workbench bundle | real art in the workbench; the decompiled source for mechanics work |
| `sound` | + `ffmpeg`; FMOD — on Windows the game's own DLLs, elsewhere the FMOD Studio API | `sts2-wiki/site/public/audio/` | sound effects in the workbench |

## Prerequisites

- **Slay the Spire 2** installed through Steam. The engine supports builds
  **v0.107.1** (stable, the default) and **v0.111.0** (beta); `import-game`
  refuses any other build unless you pass `--allow-version-mismatch` (the
  oracle and the extracted data would then describe a different game). A
  v0.111.0 install gets only its version-keyed outputs — its oracle
  (`sts2-pycli/lib/v0.111.0/`, `just build-cs v0.111.0`),
  `sts2-wiki/{extracted,decompiled,data}/v0.111.0/`, its workbench bundle
  and its art overlay (`sts2-wiki/site/public/images/v0.111.0/`: the art the
  shared tree lacks); the shared trees (`localization_eng/`, the art, the
  card db, the default workbench bundle) keep describing v0.107.1. Select the
  oracle build with `STS2_GAME_VERSION`. Steam installs one branch at a
  time; to get both builds on one machine, use `--download` (below).
- [`just`](https://just.systems), [`uv`](https://docs.astral.sh/uv/), Python 3.13.
- Rust 1.99 (pinned by `sts2-rs/rust-toolchain.toml`; rustup installs it on
  first use) and `cargo-nextest`. `sccache` is optional
  (`export RUSTC_WRAPPER=sccache`).
- **oracle tier:** the [.NET 9 SDK](https://dotnet.microsoft.com/download).
- **workbench tier:** ILSpy's CLI —
  `dotnet tool install -g ilspycmd --version 9.1.0.7988` (lands in
  `~/.dotnet/tools`; ilspycmd 10+ does not install on the .NET 9 SDK). For
  the browser build: Node 20+, `cargo install wasm-bindgen-cli --version
  0.2.120 --locked` (must match the `wasm-bindgen` crate in
  `sts2-rs/Cargo.lock`; the wasm target itself comes with the pinned
  toolchain), and `cwebp` from libwebp (`brew install webp`,
  `apt install webp`, `winget install Google.Libwebp`) — the build ships the
  art as WebP. The creature portraits are rendered with Node too
  (`sts2-wiki/tools/spine-renderer`, installed on first use by `npm ci`).
- **sound tier:** `ffmpeg` for the Opus encode (`winget install Gyan.FFmpeg`,
  `brew install ffmpeg`, `apt install ffmpeg`), and an FMOD runtime to render
  the game's sound banks:
  - **Windows:** nothing to install. The game ships `fmod.dll` and
    `fmodstudio.dll` (FMOD 2.03.06, the runtime its banks were built with)
    next to `SlayTheSpire2.exe`; `import-game` stages them beside the pck and
    renders with them.
  - **Linux / WSL:** the FMOD Studio API (free, registration-gated download
    from fmod.com — the "FMOD Engine" for Linux, `fmodstudioapi*linux.tar.gz`;
    2.03.14 is what the maintainers use). Unpack it anywhere and export
    `FMOD_SDK_ROOT=<that dir>`. The Windows DLLs cannot be used here, even
    when Steam installed the game through Proton.
  - `FMOD_SDK_ROOT`, when set, wins on every platform — if it holds that
    platform's libraries (`api/core/lib/x64/fmod.dll` on Windows,
    `api/core/lib/x86_64/libfmod.so` on Linux); otherwise the game's DLLs
    are used. On Windows, `FMOD_LIB_DIR` points at DLLs somewhere other than
    next to the pck.
  - Without ffmpeg or an FMOD runtime, `just import-game` stops before
    importing anything and says what to install. Install it and re-run, or
    use `--tier workbench` to import everything but the sound.
- **Windows:** everything here runs natively from PowerShell —
  `.\install-windows-deps.ps1` installs the tools above
  ([BUILDING-WINDOWS.md](../BUILDING-WINDOWS.md)).

## Run it

```sh
just import-game --download --steam-user NAME   # downloads BOTH builds, imports each
just import-game                       # finds the game through Steam; every tier
just import-game --tier oracle         # just the C# oracle
just import-game --tier workbench      # everything but sound
just import-game --status              # what exists, what is missing, art coverage
```

The game is found through Steam: every library folder Steam knows about
(`steamapps/libraryfolders.vdf`, so other drives too; on Windows the Steam
path comes from the registry). Anywhere else, set `STS2_GAME_DIR` to the
install folder (the one holding `SlayTheSpire2.pck`), the macOS `.app`, or the
`data_sts2_*` folder holding `sts2.dll` — or pass that folder as the first
argument (on Unix shells; on Windows the path's spaces do not survive the
recipe, so use `STS2_GAME_DIR`, e.g.
`$env:STS2_GAME_DIR = "D:\SteamLibrary\steamapps\common\Slay the Spire 2"`).

### Download the builds instead (`--download`)

`--download` fetches every supported build from Steam (`--download v0.111.0`
fetches just one) with [DepotDownloader](https://github.com/SteamRE/DepotDownloader/releases)
— on PATH, at `$DEPOTDOWNLOADER`, or `--depotdownloader PATH` — into
`sts2_binaries/<version>/`, then imports each one, the default build first
(a later build's art overlay is what the default build's art lacks). It
needs an account that owns the game: `--steam-user NAME` (DepotDownloader
asks for the password and the Steam Guard code itself; `-remember-password`
keeps its login token for the next build) or `--steam-qr` (scan a QR code
in the Steam app). The script never sees your password. Each build is pinned
to its depot manifest (`DEPOT_MANIFESTS` in `scripts/import_game.py`), not a
branch, because Steam's `public` and `public-beta` branches move on to newer
builds; only the files the import reads are downloaded (the pck and the data
folder: about 1.2 GB to download and 2.1 GB on disk per build, not the executable). Steam now and then refuses a manifest it serves a minute later, so a failed download is retried. The Windows depot is used on
every OS: the pck and the .NET DLLs are the same everywhere. A build already
in `sts2_binaries/<version>/` is not downloaded again.

The driver stages the build under `sts2_binaries/<version>/` — symlinks; on
Windows without Developer Mode a hard link for the pck and a junction for the
data folder; a copy only when neither works (the pck is ~1.9 GB, and the log
says so) or with `--copy` — and then runs, in order, calling each tool
directly (no bash, so the same on every platform):

1. `sts2-pycli/scripts/setup_oracle.py` — copies the DLLs to `lib/<version>/`,
   IL-patches `sts2.dll` for headless use (`tools/IlPatcher`), builds the
   C# harness. (`setup.sh` runs the same script.)
2. `sts2-wiki/scripts/extract_pck` and `sts2-pycli/scripts/vendor_localization.py`
   — the English localization and the atlas metadata.
3. ilspycmd (the decompile), the `sts2-wiki` extractors (`just extract`,
   `extract-vfx-bindings`, `extract-power-flash-sites` — committed, so only
   re-run when missing), then the art: `extract_images.py` (the sprite
   atlases), `extract_vfx`, `extract_rooms` (rooms, scenes, map, HUD, boss
   icons, orbs, ancients, the game's static creature sprites),
   `extract_card_template` (card frames), `extract_rig --all` (every creature
   and room rig), `render_portraits` (renders each creature rig's idle pose
   to the still `monsters/*.png` and `character_sprites/*.png` the game does
   not ship, through `tools/spine-renderer`), and `build_sfx_map`.
   A non-default build (v0.111.0) runs none of these art steps; it gets an
   art overlay instead: its sprite atlases, kept where the shared tree has
   no file (new cards, relics, potions and powers, and the final art of cards
   v0.107.1 ships only as `beta/` sketches), into `images/<version>/`, which
   that build's runs read before the shared tree. It does get its own
   `build_sfx_map` (into `sts2-pycli/web/public/<version>/sfx_map.json`).
4. `sts2-pycli`: `synergy.gen_card_text`, `workbench.build_card_db`,
   `scripts/build_asset_manifest.py`, `scripts/bundle_workbench_data.py`
   (= `just workbench-data`, the bundle the browser workbench embeds). A
   non-default build gets its own card text and card db too
   (`web/public/<version>/card_text.json`,
   `python/workbench/data/<version>/card_db.json`), but not the asset
   manifest.
5. `sound`: `extract_sfx` (renders the ~570 sound-effect events of the
   game's FMOD banks to WAV — music and ambience are not extracted) +
   `encode_sfx` (trims and encodes them to Opus/WebM).

Every step is idempotent — a re-run skips outputs that exist — and
everything is logged to `logs/import-game.log`. To redo one step, delete its
output and re-run.

At the end the driver compares the result with the committed asset
manifest — the art and sound the maintainers' tree holds — and prints the
coverage (`scripts/check_art.py`; `--list` names every missing file). A full
import covers all of it except 17 images of content the game removed or
renamed before v0.107.1 (Doormaker's boss icon, the Grapple / Follow Through /
Prepare cards, 13 powers): their art is not in the v0.107.1 pck at all, and
`check_art.py` lists them as "older builds only". If the workbench asks for
one, it draws the neutral placeholder tile.

Files present in both trees are mostly byte- or pixel-identical. Where they
differ, the reference was extracted from an earlier build and the game has
since redrawn the asset (some creature rigs, a few epoch portraits and card
illustrations, the character icons), or the portrait is a fresh render of
the current rig.

Most sound events pick a random take of each layer each time they play. The
render seeds FMOD and loads synchronously, so two imports with the same FMOD
build give byte-identical audio. The ~130 streaming events are the exception
FMOD cannot fully pin down: their streams open on a thread of FMOD's own, so
the render gives each open a fixed budget and renders every event twice,
keeping it only when both renders agree. An event whose renders never agree
(seen only under heavy load or with very slow disks) is logged as `UNSTABLE`
in `logs/import-game.log`. To redo the sound on a quieter machine, delete
`sts2-wiki/site/public/audio/` and re-run `just import-game`. Against the maintainers' audio they differ:
the same sounds with the same lengths and loudness, but different takes,
because those files were rendered before the seed existed. The Windows DLLs
(2.03.06) and the Linux SDK (2.03.14) are different FMOD builds and may pick
different takes.

Maintainers can compare a local tree with theirs:
`uv run --with pillow --with numpy --no-project scripts/check_art.py
--reference <their sts2-wiki/site/public>`.

## Then

```sh
just workbench-local                   # browser workbench: build + serve on 127.0.0.1:4173
just workbench-server                  # Python-backed workbench on 127.0.0.1:8077
cd sts2-pycli && uv sync --dev
just build-cs                          # C# oracle (oracle tier)
uv run pytest tests                    # the whole suite, oracle tests included
cd ../sts2-rs && cargo nextest run --workspace   # game-gated tests now run too
```

The workbench golden tests (`sts2-rs/crates/sts2-workbench/tests/golden/`)
ship and run as-is; `just workbench-golden-dump` (sts2-pycli) re-records them
from your data when the display text or the engine changes.

## What is pinned where

The build the engine simulates is `v0.107.1`, recorded in
`sts2-rs/crates/sts2-workbench/src/data.rs` (`ENGINE_WIKI_VERSION`),
`sts2-pycli/python/workbench/localization.py` (`_ENGINE_WIKI_VERSION`) and
`scripts/import_game.py` (`PINNED_VERSION`). `localization_eng/BUILD.txt`
records which build your localization came from.
