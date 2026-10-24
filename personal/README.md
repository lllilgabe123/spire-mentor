# Personal build of Spire Mentor

A fork of [tfriedel/spire-mentor](https://github.com/tfriedel/spire-mentor) with
changes for my own play. Everything stays **client-side**: the mod is marked
`affects_gameplay: false`, so friends in co-op need nothing installed, and no
feature changes anyone else's game.

## Changes to the mod and advisor (branch `client-side`)

- **Co-op safe:** autoplay, AI moves, undo/restart and game speed refuse to run
  in multiplayer (`SimMultiplayer.cs`); the multiplayer handshake ignores the mod.
- **Hotkeys remapped** away from other mods: Home (help), Insert (AI on/off),
  Ctrl+Shift+N (AI once), Ctrl+Shift+P (AI auto).
- **v0.111.0 card fix:** beta-only cards such as Midnight parse
  (`sts2-wire/src/state_dict.rs`).
- **Watchdog and tracker** (`SimWatchdog.cs`): runs/fights logs, "advice paused" warning.
- **Near-tie picks** (`SimTieBreak.cs`) and **risky-route warning** (`SimRouteRisk.cs`).
- **See the future in fights:** advisor `--fight-sight clairvoyant` plans with
  the true draw order and RNG (setting "See the future in fights").
- **Foresight** (`realgame_foresight.py`): lethal warning with block needed,
  enemy moves for the next turns, Feed/Hand of Greed/The Hunt kept for kills.
- **Co-op awareness** (`SimCoop.cs`): teammate damage share and suggested
  teammate turns.
- **Map room prediction** (`SimMapForesight.cs`): names the fights, elites and
  "?" rooms along the route, replayed on copies of the game's RNG.
- **Low-HP caution** engine setting `STS2_LOW_HP_CAUTION` (off by default,
  under test).

## This folder

- `tools/` - reliability tracker, watchdog, replay checks.
- `test-bot/` - the two-instance co-op test harness (`run-mp-test.ps1`).
- `training/` - the beta run-AI training campaign (Start/Stop/Status shortcuts).
- `abtests/` - simulator A/B tests.
