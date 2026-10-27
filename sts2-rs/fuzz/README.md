# sts2-fuzz

Cargo-fuzz harness for the sts2-core combat engine. Foundation
scaffolding from bead **sts2-rs-722dx** (GH #1).

## Why this lives outside the workspace

`cargo-fuzz` instruments targets with `libfuzzer-sys`, which requires
nightly Rust (`-Z sanitizer=...`) and emits an `extern "C"` shim from
the `fuzz_target!` macro. The main workspace pins **Rust 1.99.1 stable**
and sets `unsafe_code = "deny"` workspace-wide, both of which would
break the fuzz crate. So this `fuzz/` directory has its own empty
`[workspace]` table and is excluded from the parent workspace —
`cargo check --workspace` from the repo root will *not* descend in
here.

## Install

```sh
cargo install cargo-fuzz          # one-time, installs the `cargo fuzz` subcommand
rustup toolchain install nightly  # one-time, fuzzing requires nightly
```

## Build (stable check)

You can confirm the harness compiles on the pinned stable toolchain
without ever running the fuzzer:

```sh
cd fuzz
cargo build --bin combat_step
```

That's what CI / pre-commit should run.

## Run a 60-second fuzz session

```sh
cd fuzz
cargo +nightly fuzz run combat_step -- -max_total_time=60
```

Drop the `-max_total_time` to fuzz indefinitely (Ctrl-C to stop).

## Where artifacts land

- **Corpus** (interesting inputs the fuzzer discovered):
  `fuzz/corpus/combat_step/`
- **Crashes** (panics, assertion failures, timeouts):
  `fuzz/artifacts/combat_step/`

Both directories are git-ignored by convention — commit only minimized
regression seeds you want to keep.

## Triaging a crash

When libfuzzer finds a crash, it writes the input bytes to
`fuzz/artifacts/combat_step/crash-<sha>` and prints the path. Re-run
that single input deterministically:

```sh
cd fuzz
cargo +nightly fuzz run combat_step fuzz/artifacts/combat_step/crash-<sha>
```

To minimize:

```sh
cargo +nightly fuzz tmin combat_step fuzz/artifacts/combat_step/crash-<sha>
```

The crash input decodes to a `FuzzInput` (see `fuzz_targets/combat_step.rs`)
— `seed: u64`, `enemy_pick: u8`, `decisions: Vec<u16>`. The decision
stream is indices into `legal_actions(&state)` mod the action count
at each step, so the same crash input replays deterministically
against the engine.

## What it checks

After every legal action the target asserts:

- player `current_hp ∈ [0, max_hp]`
- player `block >= 0`, `energy >= 0`
- every creature `current_hp ∈ [0, max_hp]`, `block >= 0`
- `apply()` never returns `StepResult::InvalidTarget` for an action
  pulled from `legal_actions(&state)` (engine-integrity invariant)
- no panic anywhere in the engine

Each invocation is capped at **200 decisions** so a pathological
input can't wedge a single executor past libfuzzer's ~1s budget.

## Input surface

Beyond the per-character card pool the target also draws from the shared
**colorless** pool and stamps a per-card `base_replay_count` (HiddenGem's
`Replay` enchant), so a card's REPLAY iterations are reachable without the
fuzzer first having to draw and play a OneTwoPunch / HiddenGem.

Both were added while fixing GH #422 (`dispatch: empty Play pile`), which
fired from a colorless card's replay and was therefore invisible here.

## Future work

- **Seed routing / replay / auto-play powers** (Stampede, Mayhem, EchoForm,
  OneTwoPunch, Corruption, Nostalgia, Feral, Hellraiser, Stratagem, …) onto
  the player at combat start. Those powers decide where a played card goes
  and how many times it resolves, and the fuzzer assembling one by drawing
  *and* playing the right Power card is vanishingly unlikely — GH #422 lived
  behind exactly that door (a Stampede end-of-turn auto-play batch whose
  first card surfaces a select) and only fell to an ad-hoc harness that
  seeded the power directly. Do this behind an invariant-triage pass: an
  out-of-band grant is not a state the per-turn invariants have ever seen,
  so expect to sort real findings from setup artifacts first.

- Seed `fuzz/corpus/combat_step/` from real fight replays (sts2log dumps
  or the parity harness fixtures under `tests/parity/scripted/`).
- Add a second target that fuzzes individual `Card` plays in isolation
  rather than full fights.
- Wire a CI step that runs the harness for a short budget on PRs that
  touch `crates/sts2-core/src/`.
