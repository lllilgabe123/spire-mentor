# PPO champion training playbook

How to train, gate, and deploy a run-policy champion for a character without
repeating the mistakes the Ironclad campaign made. Every rule here is written
because something went wrong; the cost is noted so you can judge whether it
applies to your case rather than following it blindly.

Companion docs: `sts2-rs/docs/ppo-rust-inference.md` (the inference side and
the checkpoint-swap procedure), `docs/experiments/2026-07-30-ironclad-event-pilot.md`
(the campaign this is distilled from), and **`docs/ppo-screen-playbook.md`**
— the cheap paired screen (`ppo.screen`) and offline feature probe
(`ppo.feature_probe`) for deciding *which* idea deserves one of these
31–82-hour campaigns. A screen never promotes anything; §2 below is unchanged
and unrelaxed by it.

**Do not read champion status from this document —
`sts2-pycli/data/ppo/champions.toml` is the source of truth**, including a
one-line `gate` summary per character. A hand-kept status table used to live
here; it went stale within two weeks (it still named Necrobinder `it750` "the
only one left" after that character had been re-trained and promoted *twice* —
`asc_it300`, PR #404, then `deckmut_it150`, GH #431) and misdirected a
campaign proposal on 2026-08-20. All five characters have post-event-fix,
annealed, all-ascension champions.

**A running theme: every champion trained before 2026-07-29 has a dead
`emb_event`.** Check §0.2 before believing any prior null about that character.
Two such nulls have now been retracted — Silent's and Regent's — and both
characters gained once the env stopped hiding events.

**"High ascension is a combat-side ceiling the run policy cannot move" is dead.**
It was asserted on the strength of three nulls: Silent, Ironclad, Regent. All
three of those characters have since posted gains across the curve, and Defect —
where the same band was pre-registered as a no-regression test — took its
*largest* gains there (+18.5pp at A7, +21.0pp at A8). Stop citing it. If you
want to claim a ceiling is combat-side, measure it on an env that presents every
surface first.

---

## 0. Before you train anything: four pre-flight checks

These are cheap and each of the first three has already cost a full run
somewhere. The fourth (§0.4) is enforced by the trainer itself.

### 0.1 Is the environment actually presenting the decisions you think?

`ppo/env.py::PpoRunEnv._advance` auto-resolves any decision with a single
legal action. Anything that arrives *unbound* — needing one call to
materialise before its real options exist — gets swallowed as
`single-option-auto` and resolved at option 0 **without the policy ever seeing
it**.

That is not a hypothetical: it silently ate *every ordinary event room* until
GH #317. The meta had never once made an event choice, across every champion.

**Check, one iteration, in `<save>.stats.jsonl`:**

```
dec_per_run/event   ~6.x  = healthy
dec_per_run/event   ~1.2  = the fix is not live; you are about to waste a run
```

Generalise the habit: for any surface you expect the policy to learn, confirm
its `dec_per_run/<surface>` is non-trivial *before* committing compute.

`dec_per_run/*` carries **two granularities** (GH #399): the coarse surface
(`path`, `card-pick`, `event`, `shop`, `rest`, `treasure`) *and* the fine one
the state token uses (`card-reward`, `card-remove`, `card-upgrade`,
`card-transform`, `card-enchant`, `card-grant`, `card-smith`, `boss-relic`,
`relic-reward`, plus `unknown` — `encoder.fine_surface`'s fallback, so a
non-zero `dec_per_run/unknown` means a candidate list no kind in
`_KIND_SURFACE` claimed: investigate it, don't average it away). Read the
**fine** key for anything inside `card-pick` — all five deck-mutation surfaces
classify as `card-pick`, so a dead removal surface is invisible in the coarse
bucket next to the ~10–14/run card rewards. The two families overlap (they are
an overlay, not a partition): don't sum them, and note the coarse count wins
where a name is shared, which is why `dec_per_run/event` still means "decisions
in the event phase".

**Calibrate "non-trivial" per surface — the deck-mutation keys are legitimately
tiny.** Removals, upgrades, transforms and enchants come from seed-dependent
offers (Neow blessings, a handful of events, on-obtain relics), not from a
per-floor surface like card rewards: measured **≈0.1–0.5/run**
(`card-remove` 0.09, `card-grant` 0.06, `card-enchant` 0.19, `card-transform`
0.34, `card-upgrade` 0.47 over one 32-game iteration). Applied literally, the
"~6.x healthy" instinct from `event` would abort a perfectly live campaign. The
check for these is **non-zero over the whole iteration** — multiply back,
`dec_per_run/x × n_runs`, and expect a handful of decisions, not a handful per
run. Zero across a 64-game iteration is the abort signal; 0.09 is not.

### 0.2 Is the embedding table you care about actually receiving gradient?

Diff embedding tables between two checkpoints of the same run and take the
**median per-row drift**:

- **Uniform across all rows** ⇒ no gradient. That is weight decay
  (`--weight-decay 1e-4`) touching every row equally, not learning.
- **Per-row, concentrated in the rows you exercised** ⇒ learning.

Ironclad `emb_event` read **exactly 0.000000** across five champions, which is
what proved #317 was the root cause. Post-fix it read 0.0676 over 48/71 rows,
against `emb_relic` 0.069 / `emb_potion` 0.073. Keep an untouched table
(`emb_upg` read 0.00000) as a negative control.

Checkpoint tensors live under `ckpt["state_dict"]`, not top level.

### 0.3 Is your baseline from the same world you are measuring in?

**Honest combat search (`HiddenInfoMode::BeliefDraw`) became the default on
2026-07-12 (`f7f3cbd7c`). Measured clairvoyance rent: −22.80pp**
(McNemar p=1.1e-86, both arms one binary,
`sts2-rs/docs/experiments/2026-07-12-clairvoyance-rent-reanchor.md`).

Any win rate recorded before that date was measured by a search that could see
the true draw order. It is not comparable to anything you measure today, and
the gap is large enough to swamp most real effects.

| champion | committed | vs the flip |
|---|---|---|
| Silent `it390` | 07-09 | **before — not comparable** (retired; superseded by `asc_it300`, 08-05, after) |
| Ironclad `asc_it55` | 07-10 | **before — not comparable** (retired; superseded by `asc_it180`, 08-01, after) |
| Defect `asc_it80` | 07-11 | **before — not comparable** (retired; superseded by `asc_it230`, 08-04, after) |
| Regent `it800` | 07-25 | after (retired; superseded by `asc_it200`, 08-06) |
| Necrobinder `it750` | 07-28 | after (retired; superseded by `asc_it300`, 08-08, then `deckmut_it150`) |

Every currently-served champion is now post-flip, so the rent no longer applies
to any live number. It still applies to **historical** figures quoted in old
docs and issues — e.g. Silent `it390`'s 68.95% and Defect `asc_it80`'s 76.1%
pooled are clairvoyant-era and not comparable to anything measured today.

**Run the date check before attributing any regression to drift or staleness.**
Skipping it cost a wrong "the champion is stale" finding that had to be
retracted (`9f85701a8`) — the rent doc already warned about exactly that trap.

### 0.4 Is the trainer running on a fresh PGO wheel?

Collection is 86–97% of campaign wall clock (97% in the current 960-games/iter
recipe, 86–92% in the older 256-game legs), and nearly all of it runs inside the
`sts2_rs` extension. Its PGO build makes the same decisions as the plain build.
`sts2-rs/scripts/pgo_wheel.sh` re-checks that on every build: 50 full runs
(five characters, A0/A5/A10) must produce identical decision traces, including
each combat root's visit and value sums. It is also faster:

- **1.28×** on the trainer's own collection path: `play_batch`, `policy=net`,
  250/1000 sims, Ironclad, one pinned core, 31.2 s → 24.3 s.
- **1.33–1.35×** on the trainer-wheel workload (item M13 in
  `sts2-rs/docs/perf/2026-10-13-engine-speed-exact-design.md`).

On a 31–82 h campaign that saves hours, and the build costs about a quarter of
an hour: 14.4 min measured end to end on the dev host, plus ~3 min for the
plain build after an engine edit. Since GH #830 (2026-10-25) the default
profile workload's Ironclad runs also play the shipped Ironclad search, so the
profile covers the BC move prior in elite and boss fights. So build it first:

```bash
cd sts2-pycli
just sim-engine-pgo    # ~15-18 min; ~1 s when a fresh PGO wheel is already installed
PYTHONPATH=python .venv/bin/python -m ppo.wheel_guard   # optional: print the record the trainer will check
```

**`ppo.trainer` refuses to start** unless the imported `sts2_rs` passes three
checks, and so does `ppo.screen`, before it launches either arm:

- **Fresh.** `src_hash()` equals the hash of the on-disk source. This is the
  same content-addressed check as `parity_audit.assert_wheel_fresh`, except that
  `STS2_SKIP_WHEEL_CHECK` does not bypass it. The hash covers every workspace
  crate compiled into `sts2_rs` (engine, content, MCTS, run AI including the
  trainer's `MctsCombatPolicy`, wire, CLI loader, vocab, and the HeuristicV
  weights the run AI bakes in), the workspace `Cargo.toml`, `Cargo.lock` and the
  toolchain pin.
- **Verified.** It is not imported from one of `pgo_wheel.sh`'s own build dirs
  (`sts2-rs/target-pgo-wheel-*`, e.g. through `PYTHONPATH`) and is not the
  instrumented build. A PGO module there reports `pgo` without having passed
  the identity check (or, after `--skip-identity`, without ever running it);
  the script puts a PGO `.so` into the venv only after the check passes.
- **PGO.** `build_info()["pgo"]` is true. Only `pgo_wheel.sh`'s profile-use
  build sets that stamp, together with the sha256 of the profile it compiled
  against.

Either refusal names the fix, `just sim-engine-pgo`. The check runs in the CLI
before `train()`, so a wrong wheel costs seconds, not a collection phase.

**What keeps a PGO wheel installed, and what replaces it.** `just sim-engine`,
`just workbench`, the `sim*` recipes and anything else that goes through
`wheel-venv.sh build` keep a PGO `.so` whose `src_hash` matches the source.
They used to reinstall the plain one over it, because `maturin develop`
reinstalls even when nothing changed. `uv run` / `uv sync` (`just test`,
`lint`, `typecheck`, `check`, `advisor-warm`, …) keep it too, until a file uv
watches changes: `pgo_wheel.sh` records uv's freshness key before it builds
anything and writes that key into the dist-info when it installs the wheel
(`sts2-pycli/scripts/uv_stamp_wheel.py`, which also re-checks `src_hash`).
These replace it with a plain build, and the guard catches each at launch:

- **Any change, after `just sim-engine-pgo` started, to a file uv's
  `cache-keys` name** (`sts2-rs/crates/sts2-py/pyproject.toml`): that
  `pyproject.toml` itself, every `.rs` and `Cargo.toml` under
  `sts2-rs/crates/`, every file in a crate's `src/` (the vocab pins and the
  other `include_str!` inputs), `data/runai/heuristic_v.toml`,
  `data/prior_nets/*.json`, the workspace `Cargo.toml`, `Cargo.lock` and
  `rust-toolchain.toml`. These are the inputs
  `src_hash` covers, plus tests, benches and the other crates, so in practice
  a write to any input `src_hash` covers makes `uv run` rebuild. uv compares
  one timestamp, not contents: the newest ctime among those files on unix,
  so this includes an edit to any crate, a `git checkout` / `pull` / `stash`
  that rewrites a file, even a `chmod`; on Windows the newest mtime, which a
  restore that keeps mtimes (`cp -p`, `tar -x`, `rsync -t`) does not move.
  Deleting a file that is not the newest moves `src_hash` but not uv's key,
  so uv then keeps a wheel the guard calls stale. The trainer's guard is the
  backstop for those cases: it refuses such a wheel at launch. The next `uv run` / `uv sync`
  rebuilds a plain wheel from the current source: the stale-wheel protection
  working as intended. Nothing re-keys a PGO wheel afterwards (`just
  sim-engine` keeps it in the venv but does not tell uv), so once uv has
  replaced it, only a rebuild (~15 min) brings it back. An edit to a crate
  compiled into the wheel also moves `src_hash`, so the next `just sim-engine`
  builds plain as well. A change only uv sees (a test, the `pyproject.toml`)
  leaves the PGO wheel fresh for the guard, so `just sim-engine-pgo` is then
  a no-op although uv will replace the wheel; `WHEEL_VENV_REPLACE_PGO=1 just
  sim-engine-pgo` rebuilds it.
- **`./setup.sh` / `just build-rust`.** Both run a bare `maturin develop`.
- **`WHEEL_VENV_REPLACE_PGO=1`**, which asks for exactly that.

So run the gates (`just test`, …) first and `just sim-engine-pgo` last, right
before the launch, with no source change in between. A wheel replaced anyway
makes the trainer refuse to start; it is never trained on silently.

**A started run is pinned against a wheel swap.** The CLI copies the checked
wheel into `<save dir>/sts2_rs-pin/<sha256[:16]>/` (~30 MB, content-addressed:
pinning the same bytes again reuses the dir) and restarts itself with that dir
first on `PYTHONPATH`. The guard then checks the copy, and every iteration's
spawn workers import it, because they inherit the parent's `sys.path`.
Rebuilding or downgrading the venv's wheel mid-campaign does not reach the run
(the venv's other packages are another matter, below). Before training on a
batch, the trainer also checks that every worker reported the build the run
started on; if not, it aborts without training on that batch and names the
`--resume-from-iter` to continue from. `ppo.screen` pins once, under
`--out/sts2_rs-pin/`, for both arms and both evals. A fresh eval whose runs
report another build aborts the screen, and its file is moved aside
(`eval-<arm>.jsonl.wheel-mismatch-<time>`), so a re-run evaluates that arm
again instead of reusing it unchecked.

The pin dir is gitignored and nothing collects it; every distinct wheel a save
dir trains on adds one ~30 MB copy. **Runs whose `--save-path` shares a parent
dir share one pin root**: every default `--save-path ppo_run` launched from
`sts2-pycli/` pins into `sts2-pycli/sts2_rs-pin/`, while the recipe's
`<run>/main` gives each run its own. A live run's workers import from it every
iteration, so delete it only when no run saving into that dir is still
running, or that run's next iteration aborts. A `.pinning-*` dir in it is a
copy that a kill interrupted; delete that too.

- **The save dir must allow executing files, and be yours.** The pinned
  `.so` is loaded from there, so a `noexec` mount (some Docker volumes and
  data disks) fails the restart with an `ImportError`. Put the save dir on an
  exec mount you own: a pin dir another user created is refused, because its
  `__init__.py` would run before any check.
- **The pin covers the extension, not the Python.** Each iteration's workers
  import `python/ppo` and `champions.py` from the checkout on disk, so a
  pull, checkout or edit mid-campaign changes the code they run. Every worker
  reports a digest of those files (`py_src`), and the trainer aborts an
  iteration whose workers ran another one, naming it a source change rather
  than an sts2_rs build change. Do not pull or switch branches in the checkout
  a campaign runs from.
- **The venv itself is not pinned.** The workers run the venv's interpreter
  and its torch and numpy. In a worktree, a build after an engine edit
  (`just sim-engine`, `just sim-engine-pgo`) creates a new `.venv-h/<hash>`
  and by default deletes every `.venv-h/<hash>` no live checkout hashes to,
  including the one a running campaign started from. While a worktree
  campaign runs, build with `WHEEL_VENV_NO_GC=1`, or launch campaigns from the
  main checkout's `.venv`.
- **A resume re-pins whatever the venv holds.** The old pin cannot carry a
  resume across a source edit: the guard checks every import, a pinned copy
  included, against the current source. So after an edit, resume on a rebuilt
  wheel (`just sim-engine-pgo`), or from a checkout of the old commit to stay
  on the old engine. After a pull and a rebuild, `--resume-from-iter`
  continues on the new build. The trainer prints a
  `WARNING: iteration K's checkpoint was trained on another sts2_rs build`
  naming what moved (`src_hash`, `pgo`, the profile), and the stats rows from
  there on record the new build. A resume on other first-party Python prints
  a similar warning naming the `py_src` change.
- **Windows does not pin, and cannot build PGO.** `pgo_wheel.sh` is bash
  (Linux, or macOS with bash >= 4.3), so on Windows run `just sim-engine` and
  train with `--allow-plain-wheel`. There is no exec there either
  (`os.execve` starts a new process and exits the old one, so the shell would
  see the run end at once),
  so the CLI only runs the check and records `pinned: null`. The per-iteration
  worker check still aborts a run whose workers load another build.

**A host that cannot build PGO** (no `rustup component add llvm-tools`, Windows,
or a checkout without `sts2-rs/scripts/pgo_wheel.sh`; the refusal says which) runs
`just sim-engine` and passes `--allow-plain-wheel` to the trainer or the screen,
or exports `STS2_ALLOW_PLAIN_WHEEL=1`. The run prints a loud warning and
records that it trained on the plain wheel. Freshness has no opt-out.

**Every run records its wheel.** The startup log prints a
`wheel: PGO sts2_rs (profdata sha256 …, portable, src_hash …, pinned at …)`
line. The whole `build_info()` record (pgo, profdata sha256, target cpu,
src_hash, build_git, `.so` path, pin dir) is saved under `wheel` in every
`<save>.iter_N.pt` and in the wandb config. Each `<save>.stats.jsonl` row
carries `py_src` (the run's first-party Python digest, also in every
checkpoint) and what that iteration's workers reported: `wheel` (the build stamps plus
the `.so` they loaded, i.e. the pinned copy) and `wheel_pgo`:

```
wheel_pgo  true   = healthy
wheel_pgo  false  = trained under --allow-plain-wheel: same decisions, ~1.28x slower collection
wheel_pgo  null   = no run of that iteration reached the engine (all errors or timeouts, see
                    `errors`), so no worker reported a build and nothing was checked
```

A `wheel_pgo: false` run is not invalid, because the two builds are
decision-identical. It was only slower.

`wheel.so_path` is an absolute path, so a stats or eval file committed under
an exported path fails `just check-public`'s home-directory rule: strip it
first.

---

## 1. The recipe

The flags that actually matter. Everything else can stay at defaults. Run it
from `sts2-pycli/`, after the wheel build in §0.4:

```bash
just sim-engine-pgo   # §0.4: last, after any gates; the trainer refuses a plain or stale wheel
PYTHONPATH=python .venv/bin/python -m ppo.trainer \
  --character <Char> \
  --iterations 150 \
  --sims 250 --boss-sims 1000 --games-per-iter 960 \
  --ascension-weights "0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1" \
  --init-ckpt <prior champion>.pt \
  --entropy-coef <FROM THE PARENT — do NOT omit> \
  --entropy-decay-start <FROM THE PARENT> --entropy-decay-steps <FROM THE PARENT> \
  --entropy-coef-final 0.0125 \
  --lr-anneal-final 0.3 \
  --epochs <FROM THE PARENT — see §--epochs> \
  --save-path <run>/main --save-every 10 --device cuda
```

**`--entropy-coef` has no safe default here — resolve it before you launch.**
Omitting it takes the trainer's `0.05`, which is a *reset* to full exploration
and is wrong whenever the parent is annealed. Read the parent's own value
first:

```bash
python -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=True)['cfg']; \
  print({k:c[k] for k in ('entropy_coef','entropy_coef_final','entropy_decay_start','iterations')})" \
  <prior champion>.pt
```

(`weights_only=True` suffices — the saved cfg is a plain dict. Do not reach for
`weights_only=False` to read a checkpoint's metadata; it unpickles arbitrary
objects.)

Then apply §"Check the parent before choosing the flag" below. This line is
spelled out in the template because leaving it out is not hypothetical: the
GH #545 Ironclad campaign copied this recipe verbatim and ran 2.5× the
parent's exploration pressure (parent `simscale_it200`: `entropy_coef=0.02`,
annealed to `0.0125`), caught at it17 with `ent` climbing 0.543 → 0.816. The
playbook warned about the trap in one section and handed it to you in another.

**Inherit the parent's WHOLE schedule, not just the coefficient.** The four
entropy/epoch knobs interact, so checking one and taking template defaults for
the rest is not a partial fix — it is a different run. Print them together and
carry every one:

```bash
python -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=True)['cfg']; \
  print({k:c.get(k) for k in ('entropy_coef','entropy_coef_final','entropy_decay_start', \
  'entropy_decay_steps','epochs','iterations','policy_lr','lr_anneal_final')})" \
  <prior champion>.pt
```

This is the correction the paragraph above did not go far enough to make. The
GH #545 campaign deviated on **four** knobs, not one — `entropy_coef` 0.05 vs
0.02, `entropy_decay_start` 67% of the way in vs 20%, `entropy_decay_steps` 50
vs 30, `epochs` 4 vs 2 — because the template hardcoded the other three. It
held 2.5× the exploration bonus for 2.5× as long under 2× the gradient passes
per sample, and **failed its gate at −13.27pp** (n=1100 paired, Ironclad
A0–A10, `net_greedy`, McNemar exact p=4.6e-17) after 44 wall-clock hours. The
full post-mortem is `docs/experiments/2026-08-25-ironclad-underdocks-campaign.md`.

A deviation caught mid-run also needs re-checking against *this* list before
you reason about precedent: #545's it17 amendment found `entropy_coef`, read
it against a Silent precedent that had deviated on that flag alone, and
concluded "self-corrects, continue". It did not self-correct — because it was
not the same configuration.

### Encoder features — `export` it, do not just pass `--features`

An encoder experiment gated behind `STS2_PPO_FEATURES` (`ppo/schema.py`; screen
playbook §3) is read **at import time**, and the collection workers are spawned
fresh from the environment. `--features` does not turn anything on — it is
recorded into the checkpoint's cfg, and the trainer refuses to start when the
two disagree. So a campaign that wants one exports the var AND passes the flag:

```bash
export STS2_PPO_FEATURES=reward_upgrade   # …then --features reward_upgrade
```

The same var has to be set wherever a gated champion is **evaluated** through
the Rust arm (`run_arena`, the workbench backend, the advisor exe) —
`sts2-runai::ppo::features` mirrors it. For a feature that appends no slot,
such as `reward_upgrade`, nothing about the tensor reveals a mismatch: the
champion is simply served an observation it never trained on. Graduating the
feature at promotion (unconditional branch, gate deleted) is the way out of
depending on an env var forever.

`--device cuda` is in the block for a reason: `PPOConfig.device` defaults to
`"cpu"`, and on CPU the update costs ~497 s against ~33 s on a 3090 — the
learner goes from ~4% of iteration wall clock to ~50%. Not every champion
was trained with `cuda`: only the recent legs (Silent `cont_it300`, Ironclad
`plr_it550` and its `simscale` / `underdocks` parents) ran the update there.
The Necrobinder, Regent and Defect `deckmut_it150` champions, and Silent's
`deckmut_it150` parent, updated on the CPU (about 0.69 s per minibatch against
0.05 s on a 3090, the same minibatch size): affordable at 256 games per
iteration, about 100 s per update, not at 960. If you hand-roll a trainer
command, carry the flag.

### `--epochs 4` — adopted 2026-08-24, bundled deliberately

`epochs` sat at `2` from the day the recipe was salvaged from lightspeed,
straight through the 3.75x increase in data per update. The RLC 2026 paper
behind the collection change prescribes adding optimizer epochs when samples
per update grow; the 2026-08-22 DDR study found we had not, and that the
recipe sits well above the samples-per-KL band that produced every promoted
champion.

Mechanism verified before adoption
(`logs/2026-08-24-epochs-mechanism/FINDINGS.md`, 45 min, Ironclad, warm
start, two arms in parallel on separate GPUs):

- `nb` 642 -> 1284, **exactly 2.00x** — the knob is applied, not silently
  ignored;
- steady-state KL ratio **1.54x**, non-overlapping ranges (control
  0.0086-0.0116 across five iterations, treatment 0.0131-0.0170);
- mean DDR **5.92M -> 4.05M**, into the DDR study's stated target band;
- `clipfrac` 0.099 -> 0.127, explained variance up slightly in 5 of 5, so
  the critic is not being traded away;
- cost **+31 s on a ~900 s iteration (~3.5%)** — learner-side, so it does
  not compete for the collection thermal budget.

**What is NOT established: win rate.** The probe ran five iterations; its
0.5604 vs 0.5690 is noise and could not have been anything else. The
decision (2026-08-24) is to carry `--epochs 4` into the next campaign rather
than spend a dedicated screen or an isolated campaign on it, because
`ppo.screen` resolves only ~10pp today (~6pp if the eval panel is
quadrupled, ~4.3pp at the floor set by training divergence) and would almost
certainly return "cannot tell".

**Consequence, and it is the same one §2.5 flags for `simscale_it200`:** the
next gate measures a BUNDLE. If it promotes, the gain is not attributable to
`epochs` alone — say so in the provenance file. If you ever want the split it
costs a second campaign, so decide whether you want it *before* the gate
reads positive, not after.

**Outcome of the first campaign that carried it (2026-08-27): the gate failed
at −13.27pp**, and the bundle warning above is exactly why `epochs` cannot be
blamed or cleared. That campaign (GH #545, Ironclad) also deviated on three
entropy knobs, so its result indicts the *combination* and attributes nothing.
`epochs=4` remains **verified on mechanism, unmeasured on win rate** — the
same status it had on adoption.

The transferable mistake is not the value, it is how it travelled: a knob
adopted on mechanism alone became a hardcoded default in the template recipe,
where the next campaign picked it up without ever deciding to test it. **A
knob in that state belongs in the template as `<FROM THE PARENT>`, never as a
literal.** Until a campaign isolates it, take the parent's value: for every
champion promoted to date that is `2`.

### Collection budget — 250/1000 × 960, adopted 2026-08-20

The pre-2026-08-20 recipe collected at the eval-cheap default (1000/4000,
256 games/iter). The iso-compute experiment
(`docs/experiments/2026-08-19-training-sims-isocompute.md`, pre-registered)
compared that against 250/1000 × 960 games at equal wall clock, Ironclad,
warm-started, 75 iters, both arms evaluated at the deployed budget:

- pooled B−A **+2.50pp** (61.82→64.32%, 2,200 paired A0–A10 cells, McNemar
  p=0.028) — *below* the pre-registered 3pp bar, so the strength gain is
  suggestive, not certified (single seed-base; noise RMS ~1.3pp);
- the cheap-sims arm led at **all five** A0 checkpoints and was ~7%
  compute-handicapped;
- the same-volume high-quality control (champion + 75 more iters at
  1000/4000 × 256) gained **nothing** (+0.14pp, p=0.93) — at this stage the
  binding constraint is data volume/gradient variance, not label quality.

Adoption basis: at iso-compute, cheap-sims collection is at worst equal and
probably better, and the promotion gate (still run at the deployed budget)
bounds the downside per campaign.

**Adopt the measured point only: keep the campaign's wall clock and multiply
games, don't keep games and pocket the speedup.** The volume is the active
ingredient on current evidence, so 250/1000 × 256 (same games, 3.7× cheaper)
is unmeasured and plausibly *worse* than the old default. Two more caveats:
the evidence is Ironclad, warm-started refinement — per-character gates are
the check on transfer; and re-derive the games multiplier from the current
cost ratio when the engine changes (2026-08-19: 83.5 vs 22.5 s/run = 3.71×,
`ppo.probe` one-liner in the screen playbook §0).

**2026-10-25 (GH #830): Ironclad collection runs the shipped Ironclad
search.** The run agent's Ironclad default applies at every budget, so
collection at 250/1000 now uses the forms + combo knobs in every fight and
the BC move prior in elite and boss fights. That bundle was gated at
1000/4000 (A10) only. At 250 sims, where c_puct is 0.8, it is ungated, and
the next Ironclad campaign trains under it. Its cost is ~1.02–1.04× CPU per
combat step with the prior cache (measured at 1000/4000), which is too small
to re-derive the games multiplier. A campaign that is already running keeps
its pinned wheel (§0.4). The other four characters collect exactly as before.

### `--lr-anneal-final 0.3` — do not omit this

**The default is `1.0`, which means OFF** (`trainer.py:350` returns early). It
is silent: nothing in the log distinguishes "annealing disabled" from
"annealing working".

Omitting it cost the Ironclad pilot **~6pp**. Recovering it needed a whole
extra 30-iteration leg plus a 4400-run gate (PR #340: pooled 43.1% → 49.1%,
p=4.3e-07, positive at every ascension). Because the leg was a true resume with
LR as the only variable, that 6pp is cleanly attributable — it is a real
measurement of what the flag is worth, not an estimate.

> **That number measures a MISSING anneal, not a longer one.** Read as "an
> annealed tail is worth ~6pp" it is wrong, and the second instance settles it:
> Necrobinder's `it150 → it180` tail (2026-08-16) extended an anneal that had
> already run to completion and gated **−0.41pp on 2,200 paired cells,
> p=0.73** — a coin flip, 263 vs 254 discordant pairs
> (`docs/experiments/2026-08-16-necrobinder-annealed-tail-gate.md`). Ironclad's
> parent had `lr_frac` pinned at 1.0 for its whole run, so its tail *supplied*
> an anneal; extending a finished one buys nothing. If your parent's final
> `lr_frac` is already at its target, a tail is not the lever.
>
> The tail's training win rate rose **+1.81pp** and did not survive the paired
> sweep — one more reason §2 opens with "training win rate is not a gate".

#### Verifying it engaged — not from the first row

The obvious check does not work. `apply_lr_schedule` is
`frac = 1 − (1 − final) · it/(iterations − 1)` (`trainer.py:351`), which at
`it = 0` is **exactly 1.0** — the same value the disabled path returns. So the
first row of `<save>.stats.jsonl` reads `lr_frac: 1.0` on a *correctly*
annealed fresh run. Both Silent legs of GH #333 passed `--lr-anneal-final 0.3`
and both log `it 0 lr_frac 1.0`. A first-row check therefore raises a false
alarm on every fresh run that is configured right, and the natural reaction is
to decide the check is broken and stop running it — which is how a silent flag
stays silent.

Check the **latest** row against the closed form instead:

```bash
ITERS=300 FINAL=0.3 python3 - <run>/main.stats.jsonl <<'PY'
import json, os, sys
r = json.loads(open(sys.argv[1]).read().splitlines()[-1])
iters, final = int(os.environ["ITERS"]), float(os.environ["FINAL"])
want = 1.0 - (1.0 - final) * min(1.0, r["iteration"] / (iters - 1))
got = r["lr_frac"]
ok = abs(got - want) < 5e-4
print(f'it {r["iteration"]}: lr_frac={got} predicted={want:.4f} -> {"OK" if ok else "MISMATCH"}')
if not ok:
    print("  annealing is OFF, or --iterations differs from the value above")
PY
```

Pass `ITERS`/`FINAL` as the values you *believe* you launched with; the point is
to confront the log with an independent prediction rather than to ask the log
whether it likes itself. On Silent leg 2 at it267 this prints
`lr_frac=0.3749 predicted=0.3749 -> OK`, and reports `MISMATCH` against
`ITERS=200` — so it fails differently depending on the answer.

That second failure mode is worth catching on its own: `--resume-from-iter`
anchors to the absolute iteration, so resuming with a different `--iterations`
than the parent silently re-scales the whole remaining schedule. `lr_frac < 1.0`
cannot see that; the closed form can.

### Entropy schedule — the flat phase is load-bearing

`effective_entropy_coef` (`trainer.py:220`) holds `entropy_coef` flat until
`--entropy-decay-start`, then decays geometrically to `--entropy-coef-final`.

For a net learning a surface it has **never** exercised, the flat phase is
where it explores options it has no prior over. Ironclad's training win rate
sat at ~0.26 for 100 flat iterations and only converted (0.267 → 0.338) once
decay engaged.

> **Do not read a verdict off the exploratory phase.** I called the Ironclad
> pilot dead at it60 on exactly this mistake. It went on to +15.7pp. Then I
> predicted the anneal leg would be null and it returned +6.0pp. Then I
> predicted Defect would shrink at high ascension and it posted its *largest*
> gains there (+18.5pp at A7, +21.0pp at A8). Three mis-calls, same direction,
> same root cause: judging a run by anything other than a paired arena gate.

**The `ent` column tells you within ~20 iterations whether the flat phase has
anything to explore.** Watch it while `ent_c` is still pinned:

- **`ent` rises** under a constant coefficient ⇒ there is genuinely untrained
  surface and the flat phase is doing work. Defect `asc_it230`: 0.504 → 0.583
  flat, then sharpened to 0.429 through decay *while win rate kept rising*.
  Gated at +12.3pp.
- **`ent` sits flat** at its natural floor from iteration zero ⇒ *either* the
  net has nothing new to explore, *or* the surface you meant it to explore is
  not being presented. **These are not distinguishable from `ent` alone.**

This is an early **abort signal**, not a post-mortem — it is visible hours in
rather than days. It does not predict the *size* of a gain, and a rise is not a
verdict; only the gate is.

> **A dead-flat `ent` is a reason to re-check §0.1, not to conclude anything.**
> I originally wrote this section citing the Regent 2026-07-25 ascension leg as
> the worked example of "nothing left to explore." That was wrong. That leg ran
> Jul 25–26; the #302/#317 event fix landed Jul 29, and the leg's own stats read
> `dec_per_run/event = 1.145`. Its entropy was flat because the surface was
> **hidden**, not exhausted — and its `emb_event` drift was `0.000000` over 71
> rows, the same dead-table signature that proved #317 on Ironclad. The null was
> an artefact of the environment, and it was recorded as evidence for a
> *character-level* claim about high-ascension ceilings for a week.
>
> **The re-run settled it (2026-08-06).** On the fixed env the same character's
> `ent` *rose* 0.3137 → 0.3745 under the same pinned coefficient, then sharpened
> through decay while win rate climbed — and the leg gated **+7.5pp pooled,
> p=1.1e-10, all 11 levels positive** (`asc_it200`, GH #373). The act-1-boss
> wall the null was said to prove moved too: A9 f16 deaths 53→36, A10 54→38.
>
> **Always read `ent` and `dec_per_run/<surface>` together.** Flat `ent` *with*
> a healthy decision count is informative. Flat `ent` with a starved decision
> count tells you only that your env is broken.

> **The `ent` signal does not transfer across coefficients — state the `ent_c`
> it was calibrated at.** Both Silent legs ran at `ent_c` 0.05 and both showed a
> large `ent` rise, but leg 1 gated +14.58pp and leg 2 only +2.71pp: the rise
> tracked the *coefficient*, not the headroom. Necrobinder's `asc_it300` leg ran
> at 0.02 and its `ent` rise was correspondingly modest (0.393 → 0.422 peak),
> yet it gated +10.91pp. So a flat-ish `ent` at a low coefficient cannot
> distinguish "no headroom left" from "coefficient too low to reveal it," and the
> thresholds above are only meaningful next to the `ent_c` that produced them.
> An abort criterion carried over from a run at a different coefficient is not an
> abort criterion.

#### Conversion can lag decay onset by ~40 iterations

`--entropy-decay-start` is when the *coefficient* starts falling, not when the
*policy* starts sharpening. On Necrobinder's `asc_it300` leg, decay started at
it150 and the following 20 iterations were statistically flat — it142–171 pooled
27.67% against it80–109's 27.14%, well inside noise. I read that block as the
decay failing to convert. It was simply early: the break arrived around it200 and
ran to the end, finishing **+5.05pp (6.8σ)** above the plateau and +10.00pp above
it0–29.

**A flat block immediately after decay onset is not yet evidence of a dud.** Give
it ~40 iterations past `--entropy-decay-start` before the shape means anything,
and remember the previous rule: only the gate is a verdict.

#### Do not restart the flat phase on an already-annealed warm start

The recipe above assumes the warm start is a net whose entropy is still *high*.
When you `--init-ckpt` from a checkpoint that already finished its own decay,
`--entropy-decay-start 100` re-inflates the entropy the parent spent 50
iterations removing, and the first third of the new run goes into undoing it.

**Check the parent before choosing the flag.** Read the last row of the
parent's `<save>.stats.jsonl`:

**If that file is not on this host, read the checkpoint instead** — the `.pt`
travels with the champion, the run directory does not. The #545 campaign's
parent was trained at `~/ppo-simscale/b/main` on the other machine,
so its stats were unreachable and only the cfg (§1) answered the question:

```bash
tail -1 <parent-run>/main.stats.jsonl | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d['iteration'], d['entropy_coef'], d['ent'])"
```

- `entropy_coef` at/near the `--entropy-coef-final` that run used (the recipe's
  0.0125; the trainer's own default is 0.0025) → **the parent is annealed.** Set
  the new run's starting bonus explicitly with **`--entropy-coef`** instead of
  taking the 0.05 default.
- `entropy_coef` still at 0.05 → the parent never left its own flat phase, so
  continuing at 0.05 is continuous rather than a reset, and the recipe applies
  as written.

Setting `--entropy-coef` to the parent's exact floor buys back the wasted
iterations but also buys **no exploration**, which is the wrong trade when the
point of the run is a surface the net has never exercised. Choose an
intermediate value deliberately — and note the decay is geometric from
`--entropy-coef` to `--entropy-coef-final` (`trainer.py:224`), so lowering the
start also flattens the decay curve.

Measured on the Silent event pilot's leg 2 (`run_pilot_asc.sh`, 300 iters,
uniform A0–A10, `--init-ckpt` leg 1's it150). Leg 1 finished annealed at
`ent_c` 0.0129 / `ent` 0.521; leg 2 restarted `ent_c` at 0.05 and entropy
climbed 0.529 → **0.693** by it70–79 before reversing.

What it cost, and what it did not:

| | |
|---|---|
| acting-policy win rate, it0–9 → it60–79 | down on **10 of 11 ascensions**, A0 −22.0pp |
| **greedy** paired sweep, it80 vs the it150 warm start, 60 A0 seeds | **−6.67pp**, flips +7/−11, **McNemar p = 0.48** |

So the training log overstates the damage by roughly 3×: those are *stochastic
acting-policy* rates, and a uniformly diffuse policy loses on every ascension at
once — the same fingerprint as real degradation. The learned policy was close to
intact (a true −22pp would have put it80 near 24/60, not 33/60). Event learning
survived untouched (option-0 rate 55.4% vs the warm start's 53.1%).

It also **self-corrects**: with `ent_c` pinned at 0.05 throughout, entropy
peaked at it70–79 and fell back to 0.647 by it120 as the advantage signal grew
strong enough to outweigh the bonus, and win rate recovered with it. So this is
a waste of iterations, not a run-killer — which is exactly why it is easy to
miss. Budget ~1/3 of the run to re-annealing if you get it wrong.

Corollary for reading a warm-started run: entropy *rising* early is expected and
means the bonus is winning; entropy *turning over* is the signal that learning
has taken over. Track `ent` per iteration, not just win rate.

### Curriculum — hold it fixed across the comparison

Use **uniform A0–A10** (`0:1,…,10:1`) unless you have a reason not to, and
make it *identical to whatever produced the baseline*. If you change the
curriculum and the training fix in the same run, an A0 win-rate move is
explainable by ascension specialisation instead of the thing you were testing,
and you cannot separate them after the fact.

A0-only training is why Regent and Necrobinder are uncalibrated above A0.

### `--init-ckpt` vs `--resume-from-iter` — they are not interchangeable

| flag | loads | use when |
|---|---|---|
| `--init-ckpt <path>` | **weights only**, fresh optimizer | starting a new campaign from a prior champion |
| `--resume-from-iter N` | `<save>.iter_N.pt` + optimizer + `adv_mean`/`adv_sq` | continuing *the same* run |

For an experiment where you want to isolate one hyperparameter against a
baseline, `--resume-from-iter` is the only correct choice: `--init-ckpt`
resets Adam, which confounds "the change helped" with "resetting the optimizer
helped".

Note `<save>.pt` is stripped to `state_dict` only — a true resume needs
`<save>.iter_N.pt`, which keeps the optimizer state.

**The LR schedule is anchored to the absolute iteration**
(`t = it/(iterations-1)`, `trainer.py:344`). To append an annealed tail to a
finished 150-iteration run, declare `--iterations 180 --resume-from-iter 150
--lr-anneal-final 0.3`; the LR steps to frac 0.4134 at it150 and reaches 0.300
by it179.

---

## 2. Gating

Training win rate is **not** a gate. It is measured under exploration, on the
training distribution, without the deployed search. Use it only as a liveness
signal.

### 2.1 Rules for a valid comparison

- **One binary, both arms.** Differ only by `STS2_RUNAI_PPO`. Never set
  `STS2_RUNAI_HONEST=0`. Since GH #830 (2026-10-25), both arms of an
  Ironclad gate run the shipped Ironclad search (knobs, plus the BC move
  prior in elite and boss fights), so comparing with a pre-ship Ironclad gate
  number crosses a search change.
- **Paired seeds**, McNemar exact two-sided on the discordant pairs.
- **The full A0–A10 curve** if the net is ascension-conditioned, 200 paired
  seeds per level. An aggregate A0 number cannot clear a champion whose own
  blessing used a per-level curve, and an A0-only promotion can regress the
  high end invisibly.
- **A fresh seed panel** per gate, disjoint from every panel already spent on
  that comparison.
- **Never compare win *rates* across panels — only deltas within one.** The
  same net, weights confirmed bit-identical, scored 43.1% on one panel and
  47.2% on another (~2.9σ). Run outcomes are heavy-tailed in seed quality.

### 2.2 Do not run a cheap screen that gates the expensive curve

A screen that decides whether the curve runs is a selection filter, and it
biases whatever passes it. Either run the curve or don't. This matters most
when the expected effect is small — exactly when a screen is most tempting.

### 2.3 Pre-register the decision rule in a commit before the sweep runs

Write the promotion criteria, the panel, and the baseline into the experiment
doc and **commit it before any data exists**. The Ironclad legs did this
(`1631f2c06`, `3c97e0436`); it is what makes "every level improved" a finding
rather than a story fitted afterwards.

A rule that has already been useful:

> PROMOTE iff **(a)** pooled delta > 0 with McNemar p < 0.01, **and**
> **(b)** no single level regresses by more than 5.0pp.

#### The SHA you cite must be one that survives the merge

The whole claim is "here is the commit, made before the data existed" — so it
is worth nothing if the commit is not reachable from `main`. **It usually
isn't.** Audited 2026-08-16, of the seven pre-registration SHAs quoted across
these docs, **six were unreachable from `main`**:

- **Squash-merge destroys them.** The PR's commits never land on `main`; the
  pre-registration and the result collapse into one commit and the *ordering*
  that made it a pre-registration is gone with them. Four SHAs died this way
  (`53f040f06`, `ce84e4184`, `1631f2c06`, `3c97e0436`). GitHub's API still
  resolved some of them from the deleted branch and had already
  garbage-collected others — so this decays silently and unevenly.
- **Rebasing rewrites them.** A SHA quoted in a provenance file before a rebase
  points at nothing afterwards, in the same PR. `bb0b5b85d` → `486658148`.

Two habits fix it, and both are cheap:

1. **Merge campaign PRs with a merge commit, not a squash.** The per-commit
   ordering *is* the evidence. `gh pr merge --merge`.
2. **Re-resolve every quoted SHA after any rebase, and again after the merge**,
   then check it: `git merge-base --is-ancestor <sha> main`. If that fails, the
   citation is decoration.

The durable backstop, if the SHA is gone: `git log --follow <the gate doc>` on
`main` shows the pre-registration commit landing before the result commit —
but only if the PR was merged rather than squashed.

### 2.4 Pick the right baseline

Compare against **the currently deployed champion**, and only that. Comparing
against a two-generations-old net re-banks a gain that was already claimed and
blessed — and if that net predates 2026-07-12 you are also measuring
clairvoyance rent rather than your change (§0.3).

### 2.5 Attribution is a separate question from promotion

If the candidate beats the deployed net on a fair paired gate, it should be
promoted regardless of *why*. But say plainly which one you have.

Ironclad `asc_it150`'s +15.5pp is **still unattributed**: its baseline never
adapted to honest search, and the rent is the same order as the gain. The
`asc_it180` anneal leg *is* attributable, because it was a true resume of the
exact net it was measured against with one variable moving. Both were
promoted; only the second carries a causal claim. Write the distinction into
the provenance file so the next reader does not inherit an overclaim.

### 2.6 The standard strength eval — run it after every training

Adopted 2026-10-23. After a campaign ends (and after a promotion), run one
fixed A10 measurement at the strong combat setting, so every net has a
number on the same panel and the same settings:

```bash
cd sts2-pycli
STS2_RUNAI_MCTS_OVERRIDES="removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0" \
  OMP_NUM_THREADS=1 PYTHONPATH=python \
  .venv/bin/python -m ppo.probe --ckpt <net.pt> --policy net_greedy \
    --character <Character> --ascension 10 --act1 roll \
    --seeds 400 --seed-start 91000000 --procs 30 \
    --sims 4000 --boss-sims 16000 --run-timeout 14400 --out <cell>.jsonl
python ../docs/experiments/ironclad-a10/a10_score.py <cell>.jsonl
```

- **Settings: forms + combo at 4× sims.** `removed_counts_forms` scores
  multi-form enemies by what is left of them; the two widening constants open
  the EndTurn and draw chance nodes. 4000/16000 is 4× the deployed 1000/4000.
  Leave every other `STS2_*` variable unset, and use a fresh PGO wheel (§0.4).
- **Ironclad since 2026-10-25 (GH #830, commit `a70aa8bec`).** These three
  knobs are now the Ironclad run agent's default, so the
  `STS2_RUNAI_MCTS_OVERRIDES` line is redundant for Ironclad (harmless: it
  sets the same values). "Every other `STS2_*` unset" now also means the
  built-in BC move prior runs in Ironclad's elite and boss fights (PUCT, mix
  0.5, T 1). A wheel carries that default iff
  `sts2_rs.builtin_prior_net_json("Ironclad")` returns text. To reproduce the
  pre-ship Ironclad setting, add `STS2_RUNAI_PRIOR_NET=off`. The other four
  characters are unchanged: for them the line is still needed.
- **Panel: A10 × 400, seeds 91,000,000–91,000,399, always the same.** This is
  the one place where comparing win *rates* across nets is allowed (§2.1):
  the panel never changes, so the seed-quality variance is held fixed.
- **Report the A10 win rate and the A9 estimate** read off the same runs
  (`reached_second_boss`; `a10_score.py` prints both). The estimate ran ~5pp
  below a true A9 sweep on Ironclad.
- **Cost:** ~41 min on 30 procs, 0 errored rows in 800 (2026-10-23). On
  2026-10-26 one Ironclad arm took 38 min with the GH #830 default and 36 min
  with the prior off, again 0 errored rows in 800.
- **It is not a gate.** It runs at 4× the deployed budget. For Silent,
  Defect, Regent and Necrobinder it also measures search settings that do not
  ship: their deployed search is the BeliefDraw dialect (forms off, EndTurn
  widening C 5.787 / α 0.436, draw widening C 2.0; the 0.5 / 1.0 widening is
  `MctsConfig::default`, which that dialect replaces). For Ironclad the
  settings are the shipped ones (see above). Promotion still follows §2.1–2.3
  at 1000/4000.

**Why these settings.** On Ironclad `a10_it700`, paired on this panel (a
private eval folder, `~/ppo-ironclad-a10/evalmode/`, not published): forms +
combo at 4× scored 23.0% A10 / 49.8% A9-est; defaults at 4× scored 21.5% /
47.2%. The +1.5pp is not significant (discordant 43/37, p = 0.58), and on
`plr_it550` at 1×, forms + combo was flat (+0.25pp A10, −2.0pp A9-est). It
costs only ~7% more wall time, so it was adopted as the strongest setting
measured. 16× sims (16000/64000) added only +1.5pp over 4× for 4× the cost, so
the budget stops at 4×.

**Reference numbers** (add a row per champion):

| net | A10 | A9 estimate | date |
|---|---|---|---|
| Ironclad `plr_it550` (older wheel, pre-#771; pre-bundle, no prior) | 17.0% | 45.2% | 2026-10-21 |
| Ironclad `a10_it700` (pre-bundle, no prior) | 23.0% | 49.8% | 2026-10-23 |
| Ironclad `a10_it700`, GH #830 default (knobs + BC prior in elite/boss; PGO wheel `4e5bb730d28e2976`) | 26.75% [22.65, 31.29] | 52.25% [47.36, 57.10] | 2026-10-26 |
| Ironclad `a10_it700`, same wheel, prior off (`STS2_RUNAI_PRIOR_NET=off`) | 24.00% [20.07, 28.42] | 49.75% [44.88, 54.63] | 2026-10-26 |
| Ironclad `cardmods_it850` candidate (`card_mods,reward_upgrade`; GH #830 default; PGO wheel `aa7acaa26a2bf981`) | 30.00% [25.72, 34.66] | 56.50% [51.60, 61.27] | 2026-10-27 |

The first two Ironclad rows ran the knobs from the env line and no move
prior, on older wheels; the 2026-10-23 row also ran on another host. The last
two ran on one wheel and are paired by seed. Their brackets are Wilson 95%
CIs at n = 400, about ±4.3pp at A10; `a10_score.py` prints the rates to one
decimal (26.8% / 52.2% and 24.0% / 49.8%).

Compare a new Ironclad net with the GH #830 default row: it is what the
command above measures now, with every other `STS2_*` unset. Run the new net
on a wheel that carries the default (`builtin_prior_net_json("Ironclad")`
returns text), and pair it by seed instead of comparing rates. On `src_hash`
`4e5bb730d28e2976` (`.so` `823fa393…`), pair it with the default row's own
rows: `python ../docs/experiments/ironclad-a10/a10_score.py
../docs/experiments/2026-10-26-ic-a10-it700-standard-eval/runs/D.jsonl
<new>.jsonl`. A build from the public snapshot of the same engine source
reports `src_hash` `6a98001ae326637b` (the export strips C# quotes from
comments) and replayed both arms' rows exactly on every run checked: 24 runs
on 14 seeds, listed in the write-up. On any other wheel, re-run `a10_it700`
with the default on that wheel (~38 min on 30 procs on a PGO wheel) and pair
with that run. A rate compared across wheels mixes engine changes
with run noise at the size of the deltas this eval is read for.

The prior-off row is the pre-ship setting (knobs only) on that same wheel,
which also has PR #940's rule 5c. It is consistent with the 2026-10-23 row
(unpaired: 1.0pp above it at A10, level with it on the A9 estimate to the
printed decimal). Paired, the default scored +2.75pp over prior off at A10
(48 seeds won only with the prior, 37 only without; exact McNemar p = 0.278)
and +2.50pp on the A9 estimate (55/45, p = 0.368). At n = 400 this panel
cannot tell that from zero. The ship rests on the two A10 gates at 1000/4000.
The write-up, the per-seed rows, the driver and the analysis scripts are in
`docs/experiments/2026-10-26-ic-a10-it700-standard-eval.md` and its sibling
folder, which the public snapshot carries too. Its Reproduce section runs the
eval from a fresh public clone; the snapshot ships no `.pt`, so it rebuilds the
`a10_it700` checkpoint from the shipped `.onnx` first.

---

## 3. Promotion — one manifest, one command

`sts2-pycli/data/ppo/champions.toml` is the source of truth for which net each
character serves. Promote with:

```
cd sts2-pycli
just promote-champion <character> <onnx-filename>
```

That rewrites the character's `onnx` / `ckpt` / `sha256` in the manifest and
regenerates the two derived regions. `just check-champions` is the gate (it
also runs inside `just check` / `just ci`).

Who reads the manifest:

| consumer | how |
|---|---|
| `workbench.ppo_policy` (both backends) | reads it at runtime, via `python/champions.py` |
| `packaging/SimAdvisor.spec` (advisor exe) | reads it; ships it into the bundle |
| `synergy/embed_geometry.py` (taxonomy view) | reads it |
| `sts2-rs/…/ppo/net.rs` `champion_onnx_filename` | **generated** region |
| `./Dockerfile` `COPY` block | **generated** region |

The last two cannot read a TOML file — the Rust crate is built from standalone
worktrees with no sibling `sts2-pycli/`, and `COPY` is Dockerfile syntax — so
`scripts/sync_champions.py` writes them and `--check` byte-compares. Never
hand-edit inside the markers.

### Why this replaced a four-list checklist

The old rule was "update Rust, Python and the Dockerfile, and hope you did".
Worse, the Python resolver *globbed*: `_newest(stem)` took the highest `<N>`
**present in the tree it ran in**, and the repo, the Cloud Run image and the
frozen advisor bundle are three different trees. So one function returned three
different champions and never said so.

- PR #339 promoted Ironclad `asc_it150` and left the `COPY` on `asc_it55`. The
  container served a pre-honest-search net until #340. Nothing failed loudly:
  every list was self-consistent and the image was valid.
- Retired nets stay committed as paired-gate baselines, and several have a
  *higher* iteration than the champion that replaced them (Necrobinder `it750`
  vs `asc_it300`, Regent `it800` vs `asc_it200`). Under the glob, such a file
  outranked the real champion wherever both were present.
- #404 promoted Necrobinder fifteen minutes before #402 merged with the spec
  still naming `it750`, which still exists — the existence check would have
  passed and shipped the retired net.

Naming the exact file kills that class outright rather than detecting it: a
stale net on disk is inert, and a *missing* champion is a loud error instead of
a quiet demotion. The `sha256` field closes the other half — it enforces "ship
the exact `.onnx` the gate swept, don't re-export", which nothing checked
before.

`sts2-pycli/tests/test_champion_resolvers_agree.py` covers the generated
regions, the hashes, `.pt`/`.onnx` iteration agreement, and the retired-net
inertness.

### The taxonomy view follows the champion now

`synergy/embed_geometry.py` reads a champion's `net.emb_card` table to build the
card taxonomy (spec §5 View 3). It serves nobody, so drift there costs a stale
*analysis*, not a stale deployment — which is exactly why nobody noticed it
naming retired Ironclad and Defect nets, and no Regent net at all, across
several promotions.

It kept its own hand-edited table; #411 made it delegate, and it now resolves
from the manifest like everything else. **A promotion therefore re-points it
immediately** — so re-run any analysis whose numbers were computed against the
outgoing net's taxonomy. `just promote-champion` prints that reminder.

### What deliberately does NOT follow the champion

`ppo/dump_parity_fixtures.py`'s `--ckpt` default and the tests pinned to
recorded parity fixtures. A fixture is a recording of one specific net;
re-pointing it silently invalidates the comparison it exists to make.

### Full swap checklist

Follow `sts2-rs/docs/ppo-rust-inference.md` §Checkpoint-swap procedure, then:

- [ ] `data/ppo/<stem>_it<N>.{pt,onnx,provenance.md}` committed
- [ ] **Ship the exact `.onnx` the gate swept** — copy it, don't re-export.
      Then the deployed artifact is bit-identical to the measured one.
- [ ] No `.onnx.data` sidecar. `external_data=False` is load-bearing; without
      it dynamo splits the weights and the tract loader does not resolve them —
      the arm runs on garbage rather than failing.
- [ ] `just promote-champion <character> <file>` run, then the `gate` line in
      `champions.toml` updated by hand (it still describes the OLD net)
- [ ] `just check-champions` and `test_champion_resolvers_agree.py` green
- [ ] Parity fixtures regenerated (pass `--ckpt` explicitly)
- [ ] `cargo check` / `clippy -D warnings` / `fmt` / `nextest --workspace`
- [ ] The 4 Rust PPO gates and the 2 pytest parity gates
- [ ] **Release smoke with `STS2_RUNAI_PPO` unset**, so it exercises the
      *resolver* path rather than the env override the gate used. 5 seeds,
      expect all runs to complete with no panic.
- [ ] Overflow headroom checked (below)
- [ ] **Published win rates updated** if the headline numbers moved: the
      public README's win-rate table (`release/public-overlay/README.md`) and
      the project's external directory entry,
      [slaythespire.dev/projects/spire-mentor](https://slaythespire.dev/projects/spire-mentor/)
      — a PR to `src/content/projects/spire-mentor.md` in
      [`ptrlrd/slaythespire-dev`](https://github.com/ptrlrd/slaythespire-dev)
      (first added in its PR #8, 2026-10-26). Say which runs a number comes
      from (n, sims, date), as that entry does.

### Overflow headroom

Fixed graph shapes: `T_MAX=320` tokens, `A_MAX=160` action slots. **Overflow
is fatal, never truncated** — the policy panics rather than score a decision
with options hidden from it. The live binder is `obs + act ≤ T_MAX` at
deck ≈ 126.

Every sweep row carries `ppo_peak_act_slots` / `ppo_peak_tokens`, so mine your
gate's thousands of runs rather than trusting a 5-seed smoke. Ironclad
`asc_it180` over 2200 runs: peak 75/160 slots, 199/320 tokens, max deck 61,
zero panics.

---

## 4. Operational gotchas

Each of these produced a wrong or empty result at least once.

- **`STS2_RUNAI_EVENT_BLEED_LOG` fails open.** Run a sweep from a checkout
  that lacks the probe and you get a normal-looking arena report with **zero**
  bleed rows. Assert `test -s` on the log; check the branch is the one with
  the probe.
- **`run_arena sweep` writes its report to `--out`, not stdout.** A `>` redirect
  yields an empty file and a silently discarded sweep.
- **`pgrep -f <pattern>` matches your own wrapper's cmdline.** It reported a
  finished job as still running twice. Use the bracket trick:
  `pgrep -f "[m]y_script.sh"`.
- **Poll a logfile sentinel, not a process**, for long jobs; arm one
  fire-once waiter rather than a per-cell monitor.
- **A guard you have not seen fail is not a guard.** Verify RED against the
  real bug before trusting it (§3's resolver test was checked this way).
- **Save intermediate results.** Per-iteration `stats.jsonl` and per-cell
  sweep files with skip-if-exists let you see a stuck run early and resume a
  killed gate without redoing it.

---

## 5. Worked example

The Ironclad campaign end to end, as a template:

| leg | what | result |
|---|---|---|
| root cause | `emb_event` drift exactly 0.000000 across 5 champions | GH #302 → fix #317 |
| pilot | 150 iters, uniform A0–A10, warm start `asc_it55` | events 1.22 → 6.66 dec/run; bleed 1203 → 640; lethal-vetoes 6 → 0 |
| gate | A0–A10 × 200 paired seeds, panel 7000–7199 | 31.6% → 47.2%, every level positive → promoted (PR #339) |
| polish | +30 iters, `--lr-anneal-final 0.3`, true resume | 43.1% → 49.1%, p=4.3e-07 → promoted (PR #340) |
| deploy | Dockerfile `COPY` was still on `asc_it55` | caught in #340; guard test added |

Two predictions I got wrong, both by judging a run on something other than a
paired arena gate. If your intuition disagrees with a gate, the gate is right.
