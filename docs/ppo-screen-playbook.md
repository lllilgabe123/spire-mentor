# PPO screening playbook

How to find out whether an idea is worth a campaign, in under an hour instead
of 31–82.

Companion to `docs/ppo-champion-training-playbook.md`, which is still the only
document that governs **promotion**. This one governs **triage**. The two are
not interchangeable and the difference is the whole point:

> A screen decides *what to spend 31 hours on*. It never decides what to ship.

---

## 0. Why a screen, and what it costs

Measured from the committed `sts2-pycli/data/ppo/*.stats.jsonl`:

| campaign | iters | total | collect | train | s/iter |
|---|---|---|---|---|---|
| Regent `asc_it200` | 150 | **30.97 h** | 26.79 h (87%) | 4.18 h | 743 |
| Necrobinder `asc_it300` | 300 | 62.64 h | 54.51 h (87%) | 8.12 h | 752 |
| Silent `asc_it300` | 300 | 81.99 h | 73.31 h (89%) | 8.68 h | 984 |

All at `games_per_iter=256`, combat at `CombatBudget::eval_cheap()` = 1000 sims
/ 4000 boss sims (`sts2-rs/crates/sts2-runai/src/policy/mod.rs:162`).

Since 2026-10-25 (GH #830) an Ironclad screen runs the shipped Ironclad search
in both arms: the forms + combo knobs in every fight and the BC move prior in
elite and boss fights. The other four characters search as before. A screen
delta is still a within-panel comparison, but an Ironclad screen run before
that date and one run after it measured different searches.

**Collection is ~87% of wall clock and combat search is nearly all of it.** The
rest is the PyTorch update, which used to pin the floor at ~4 h and no longer
does — `--device cuda` moves it off the critical path.

### The measured sims → cost curve

Single core, `Ironclad`, A0, always-take-option-0 policy (so the absolute
numbers are optimistic on run length; the *ratios* are the point):

| backend | s/run | s/floor | speedup |
|---|---|---|---|
| MCTS 1000 / 4000 (campaign) | 18.5 | 0.84 | 1× |
| MCTS 250 / 1000 | — | — | ~4× (interpolated) |
| MCTS 100 / 300 | 1.80 | 0.069 | **12×** |
| MCTS 32 / 100 | 0.57 | 0.025 | **33×** |
| surrogate (stub net) | 0.073 | 0.011 | ~75× |

Re-measured 2026-08-19 (`ppo.probe --policy net`, deployed champion, A0, 64
seeds, post-#467/#469 widening — absolute costs are ~4.5× the table above
because the stochastic champion reaches deep acts and the deployed search
widens; the near-linear *shape* holds):

| backend | mean s/run | speedup | win rate (64 seeds, context) |
|---|---|---|---|
| MCTS 1000 / 4000 | 83.5 | 1.00× | 71.9% |
| MCTS 250 / 1000 | 22.5 | **3.71×** | 60.9% |
| MCTS 100 / 300 | 8.5 | 9.8× | 53.1% |
| MCTS 32 / 100 | 3.8 | 22× | 20.3% |

The 250/1000 row is no longer only a screen setting — it is the campaign
collection default at iso-compute (960 games/iter); see the champion
playbook §1 and `docs/experiments/2026-08-19-training-sims-isocompute.md`.

**Cost falls almost linearly with sims — there is no fixed per-run floor in
this range.** That is the finding the whole design rests on, and it means the
cheap-MCTS screen gets you most of the way without the surrogate: dropping to
32/100 is within ~2× of replacing combat entirely.

Re-measure this curve when the engine changes. The one-liner:

```bash
cd sts2-pycli
for S in 1000:4000 250:1000 100:300 32:100; do
  PYTHONPATH=python .venv/bin/python -m ppo.probe --policy net \
    --ckpt data/ppo/<champion>.pt --seeds 32 --procs 46 \
    --sims ${S%:*} --boss-sims ${S#*:}
done
```

### What the screen gives up

Four deliberate distortions, each of which can flip a verdict:

1. **Weaker combat** changes which decks are good. A card that only pays off
   with good play may look worthless at 100 sims.
2. **Fewer games/iter** means noisier gradients.
3. **Fewer iterations** means the screen sees the early part of a learning
   curve. The Ironclad campaign's training win rate sat flat at ~0.26 for
   **100** iterations before converting — a screen would have called it dead,
   as the playbook's author did in real life.
4. **A narrower ascension band** (A0–A4) drops exactly the levels where several
   campaigns took their largest gains.

Distortion 3 is the dangerous one, and it is why §4 exists.

---

## 1. Tools

| tool | question it answers | cost |
|---|---|---|
| `ppo.feature_probe` | does this feature carry outcome information the encoder lacks? | minutes |
| `ppo.shaping_gate` | can the PBRS shaping coefficients move the gradient at all? | seconds |
| `ppo.screen` | does this change move a paired win rate in a cheap world? | ~1 h |
| a campaign + `docs/ppo-champion-training-playbook.md` §2 | should we ship it? | 31–82 h |

Work down the list. Each stage is a filter, not a verdict, until the last.

### 1.1 `ppo.feature_probe` — offline, minutes per idea

Record a corpus once, reuse it forever:

```bash
cd sts2-pycli
PYTHONPATH=python .venv/bin/python -m ppo.probe --policy net \
  --ckpt data/ppo/<champion>.pt --record --seeds 2000 --procs 46 \
  --out /tmp/ppo-corpus/<character>.jsonl
```

Then per candidate feature:

```bash
PYTHONPATH=python .venv/bin/python -m ppo.feature_probe \
  --corpus /tmp/ppo-corpus/<character>.jsonl --feature boss \
  --repeats 3 --out /tmp/probe-boss
```

It fits the run policy with and without the feature on the frozen corpus and
compares **held-out value-head explained variance**.

**Read it correctly:**

- A gain is **necessary, not sufficient**. Information about the return is not
  the same as information that changes the argmax.
- `policy_ce` is expected to be **flat**. The champion that generated the
  corpus could not condition on the feature, so its actions carry no signal
  about it. A flat CE is not a null. A *degrading* CE means the extra table is
  costing capacity, which is worth knowing.
- The probe refuses to conclude when the corpus cannot support one — fewer than
  2000 held-out decisions, or a feature that is near-constant in the corpus. If
  it prints `*** INCONCLUSIVE ***`, do not file a null.

### 1.2 `ppo.screen` — paired, ~1 h

```bash
cd sts2-pycli
just sim-engine-pgo   # after any gates; the screen refuses a plain or stale wheel
PYTHONPATH=python .venv/bin/python -m ppo.screen \
  --name boss-id --character Ironclad \
  --init-ckpt "data/ppo/$(<the ckpt named for this character in data/ppo/champions.toml>)" \
  --arm-b-features boss \
  --device cuda --procs 46 \
  --out ~/screens/boss-id
```

**The screen trains on the PGO wheel only**, like a campaign (champion playbook
§0.4). It checks the imported `sts2_rs` before it launches either arm, so a
plain or stale wheel fails in seconds rather than one arm in. Each arm's
trainer checks again. The PGO build makes the same decisions as the plain one
and collects ~1.28× faster. `just sim-engine-pgo` costs ~15–18 min (14.4 min
measured on the dev host with the plain build current) and is a no-op when
the installed wheel is already fresh. `uv run` (`just test`, …) keeps it until
a file uv's cache-keys name (`sts2-rs/crates/sts2-py/pyproject.toml` itself,
any crate's `.rs`, `Cargo.toml` or `src/` file, `data/runai/heuristic_v.toml`,
`data/prior_nets/*.json`, the workspace `Cargo.toml`, `Cargo.lock`,
`rust-toolchain.toml`) changes after
the build started, even by a ctime alone; after that, a rebuild is the only
way back, so build it last (champion playbook §0.4 lists what replaces it).
A pgo_wheel.sh staging or `--skip-identity` build put on `PYTHONPATH` is
refused, even with `--allow-plain-wheel`. On a host that cannot build PGO (no llvm-tools, or
Windows, where the bash build does not run), pass `--allow-plain-wheel`: the screen exports `STS2_ALLOW_PLAIN_WHEEL=1`, both
arms' trainers inherit it, and `screen.json` plus each arm's `stats.jsonl`
(`wheel_pgo`) record it. `--dry-run` launches nothing and skips the check.

The screen **pins** the checked wheel once: it copies it to
`--out/sts2_rs-pin/<sha256[:16]>/` and restarts itself with that dir first on
`PYTHONPATH`, and both arms' trainers and both evals import that copy. So the
two arms cannot train on different wheels, and rebuilding the venv while a
screen runs does not reach it. Each eval row records the build its run
loaded, and a fresh eval whose runs report anything but the pin aborts the
screen and is moved aside (`eval-<arm>.jsonl.wheel-mismatch-<time>`), so the
next invocation evaluates that arm again. Eval runs that ended in an error or a
timeout are left out of the pairing (a crash is not a loss) and counted as
`dropped_<arm>` in `report.json`. An arm reused from an earlier invocation (its `main.pt` exists)
trained on whatever wheel that invocation had; `screen.json` lists it under
`wheel.reused_arms` with the build its last iteration's workers reported, and
a reused `eval-<arm>.jsonl` under `wheel.reused_evals` with the build its runs
reported (null for a file that predates the record). The pin dir stays after
the screen; delete it when done. `--out` must be a directory you own (a pin
dir another user created is refused) on a mount that allows executing files,
since the pinned `.so` loads from there; a shared `/tmp` is neither. On Windows there is
no pin (no exec; champion playbook §0.4): the screen and each arm's trainer
check the venv's wheel at launch, and the per-iteration and eval checks are
what catch a build that changes under them.

Both arms share `--seed-base`, so the trainer's
`seed_base + it*games_per_iter + g` derivation and its seed-derived ascension
draw hand them an **identical seed × ascension panel every iteration**. The
eval panel is shared too, so the headline is a McNemar over discordant seeds
rather than a difference of two independent rates.

Non-feature changes go through `--arm-b-flag KEY=VALUE`:

```bash
  --arm-b-flag lr_anneal_final=1.0     # arm B without the anneal
```

**The preset** (`ppo/screen.py::PRESET`, all overridable):

```
--sims 100 --boss-sims 300 --games-per-iter 128 --iterations 80
--ascension-weights "0:1,1:1,2:1,3:1,4:1" --lr-anneal-final 0.3
--entropy-coef 0.02 --entropy-decay-start 30 --entropy-decay-steps 30
--entropy-coef-final 0.005
```

**Always `--init-ckpt` from the deployed champion.** 80 short iterations cannot
reach from-scratch learnability; without a warm start you measure which arm
gets off the floor faster. The driver warns if you omit it.

**Read the liveness signals, not just the delta.** The report prints
`dec_per_run/*`, `ent`, `ent_c` and `explained_variance` for both arms, because
champion-playbook §0.1 applies unchanged: a null against a starved surface says
nothing about the change. A `dec_per_run/event` near 1.2 means the env is
broken, not that the idea failed.

### 1.3 `ppo.shaping_gate` — offline, seconds (GH #464)

Answers a narrower question than the other two, and answers it decisively:
**can the PBRS shaping coefficients in `ppo/rewards.py` move the gradient at
all?** Reuses the same corpus as `ppo.feature_probe`, so it is free once you
have one.

```bash
cd sts2-pycli
PYTHONPATH=python .venv/bin/python -m ppo.shaping_gate \
  --corpus /tmp/ppo-corpus/<character>.jsonl \
  --calibrate-ev 0.65 --json /tmp/shaping-<character>.json
```

**Do not send a shaping change to `ppo.screen` first.** It cannot resolve one,
and the reason is not the ~8 pp noise floor of §4 — it is that the effect is
*zero* on the screen's axis. With γ=1 and terminal shaping un-credited, take
any critic of the form `V(s) = m(s) − Φ(s)`, which is the form the fitted
critic converges to:

```text
delta_t = r_t + V(s_{t+1}) - V(s_t)
        = (Phi_{t+1} - Phi_t) + (m_{t+1} - Phi_{t+1}) - (m_t - Phi_t)
        = m_{t+1} - m_t
```

Every `Φ` cancels, so **every TD residual — and therefore `A_lam` at every
λ, not just λ=1 — is independent of the coefficients.** This is stronger
than the policy invariance the `rewards.py` docstring cites (Ng, Harada &
Russell 1999), which only fixes the optimal policy: here the gradient
*estimator* is untouched. Shaping can act only through **critic misfit**.

So the tool reports two tables:

- **ideal critic** (`V = m_hat − Φ`) — must be flat to float tolerance. It is
  the executable form of the identity above and a self-check on the corpus; if
  it is not flat, the run is void and the exit code is 2.
- **fitted critic** (ridge on the shaped return, k-fold *by run*) — the
  realistic one, which has to spend capacity cancelling `Φ`. Its `adv_corr`
  spread is the answer.

`--calibrate-ev` sets the ridge penalty so the fitted critic's `ev_return`
matches a campaign's logged `explained_variance` (0.34–0.65 across the
committed `*.stats.jsonl`), so the operating point is measured rather than
assumed. It only ever misses **low** — a penalized-linear basis cannot reach a
transformer's EV, and a small corpus reaches less. A worse critic has more
misfit and misfit is the only channel, so **a large calibration gap makes the
reported spread a ceiling, not an estimate**. The tool says so when it happens.

Read it as:

| `adv_corr` spread | what it means |
|---|---|
| < 1e-3 | shaping cannot move the gradient here. Standardize the knob; do not spend a screen arm. |
| > 1e-2 | worth **one** screen arm, read on iterations-to-a-fixed-win-rate — not terminal win rate, which is invariant by construction. |

Two things it cannot do. The ranking among candidates is a statement about a
penalized-linear critic, whose misfit is a ridge penalty; the real critic is a
transformer whose misfit is dominated by optimizer lag. And it never promotes
anything — champion-playbook §2 is unchanged by it. Take the magnitude, leave
the winner.

### 1.4 The 4×3090s

The net is small (dim 192, 3 layers) — data-parallel training buys nothing.
Spend the GPUs on:

- `--device cuda`, which removes the update from the critical path entirely;
- **arm concurrency**: run four different ideas overnight. The binding
  constraint is the 46 collection cores, so concurrent screens split cores and
  each runs proportionally slower — concurrency raises *throughput of ideas*,
  not the latency of one comparison.

---

## 2. Surrogate combat

> **CLOSED, 2026-08-14 — read
> [`combat-surrogate-investigation.md`](combat-surrogate-investigation.md)
> before doing anything in this section.** A surrogate cannot carry a PPO
> training loop: on a PPO-driven Acts 1–3 corpus a surrogate-driven agent ends
> runs 34–50% of max HP too healthy and survives at 0.64–0.68 against a
> reference 0.29. Per-fight accuracy stays good throughout (hp MAE 0.071 vs a
> lookup table's 0.135), so **the metrics in this section cannot see the
> failure** — only a compounded, closed-loop trajectory can
> (`python/fidelity_combat_surrogate.py`). Eight hypotheses were tested and
> eliminated, including everything §2.1 recommends. The sections below are kept
> for the harvest/gate mechanics, which are still correct and still used.

`--combat surrogate --surrogate-path <net.json>` replaces the per-decision
combat MCTS with one forward pass per fight
(`RunManager::surrogate_resolve_combat` → `sts2_runai::planning::
fast_forward_combat` → `RunManager::resolve_combat_fast`). Rewards are
unaffected: the fight is stamped terminal and the ordinary
`exit_combat_and_transition` generates gold / potions / card rewards exactly as
a played fight would.

**The net is not committed.** It has to be harvested and trained per character
(`combat_surrogate_data --realistic` → `train_combat_surrogate_net.py` →
`eval_combat_surrogate_net.py`).

### 2.1 Gate it before believing it — deck sensitivity

`eval_combat_surrogate_net.py` now prints a **within-`(encounter, ascension)`
rank agreement** section. This is the gate that matters for screening:

> The M1 eval (`sts2-rs/docs/experiments/2026-07-01-surrogate-v107-eval.md`)
> found the surrogate barely beat a per-`(encounter, ascension)` lookup table on
> win Brier (0.0157 vs 0.0198) and **lost** to it on HP MAE (6.29 vs 5.66), on
> data where 94% of rows sat above `win_rate` 0.9.

A headline AUC of 0.989 cannot distinguish "knows this deck is strong" from
"knows this encounter is easy". Removing the bucket mean can. If the win head's
within-bucket Spearman rho is below `DECK_SENSITIVITY_MIN_RHO` (0.20), the
surrogate is close to a lookup table, and **surrogate-world combat presents no
deck-quality gradient** — screening a card-pick change against it measures
nothing. In that case restrict it to path / event / shop / map questions and
screen deck-shaped changes at low MCTS sims instead.

That threshold was set from first principles, not from a measurement. **It has
now been measured, and the M1 conclusion does not survive.**

Measured 2026-08-12, Ironclad, 48,000 labeled combats (`combat_surrogate_data
--seeds 4000 --variants 12 --k 12 --sims 64`), 9,600 held-out rows split by
seed over all 305 `(encounter, ascension)` cells:

| model | win rho | hp rho | win MAE | hp MAE |
|---|---|---|---|---|
| `bucket_mean` (lookup table) | — (degenerate) | — | 0.1989 | 0.1321 |
| `mlp` (the shipped architecture) | 0.6617 | 0.7503 | 0.0875 | 0.0698 |
| transformer (deck as tokens) | 0.6767 | **0.7556** | 0.0774 | **0.0522** |

On *synthetic* rows the architecture looks irrelevant: the transformer beats
the shipped MLP by +0.005 hp rho, inside noise. **That conclusion does not
survive contact with realistic data.**

Same gate, same code, on a `--realistic` harvest (18,603 rows from driven
self-play, 3,605 held out over 22 qualifying cells):

| model | win rho | hp rho | win MAE | hp MAE |
|---|---|---|---|---|
| `bucket_mean` (lookup table) | — | — | 0.0557 | 0.0512 |
| `mlp` (the shipped architecture) | 0.4562 | **−0.0153** | 0.0488 | **0.0715** |
| transformer (deck as tokens) | 0.4989 | **0.2789** | 0.0277 | **0.0459** |

**On realistic states the shipped MLP has NO within-cell HP ranking at all
(rho ≈ 0) and is WORSE than the lookup table on HP MAE (0.0715 vs 0.0512).**
That is the M1 finding, reproduced exactly — "barely beat a lookup table on
win, and *lost* to it on HP MAE". M1 was right about the net it tested. The
transformer clears the gate on the same data (hp rho 0.279) and beats the
lookup table on both MAEs.

So the honest reading is the opposite of what synthetic rows suggest:

> Synthetic decks are easy enough that a bag-of-cards MLP suffices. Realistic
> decks are narrow, correlated and saturated, and there the MLP collapses to a
> lookup table while a transformer does not. **Benchmark surrogates on
> `--realistic` rows only.** A synthetic-only PASS is close to meaningless —
> it was on synthetic rows that the MLP looked fine.

### 2.1.0 The recipe: train on realistic + synthetic, score on realistic

The realistic-only numbers above are **data-starved**, and fixing that changes
the conclusion. Mixing synthetic rows into *training* while scoring on the same
realistic holdout (3,605 rows, 22 cells) gives a clean dose-response:

| synthetic added to training | model | win rho | hp rho | hp MAE |
|---|---|---|---|---|
| none (14,998 realistic) | mlp | 0.4562 | −0.0153 | 0.0715 |
| none | transformer | 0.4989 | 0.2789 | 0.0459 |
| +15k (26,998) | mlp | 0.4876 | 0.2433 | 0.0616 |
| +15k | transformer | 0.5099 | 0.3193 | 0.0438 |
| +48k (53,398) | mlp | 0.5039 | **0.4622** | 0.0522 |
| +48k | transformer | **0.5230** | **0.5237** | **0.0371** |

Monotone in both models and still climbing at 48k. So:

* **Harvest both. Train on the union, score on realistic only.**
  `train_combat_surrogate_transformer.py --labels <realistic>
  --labels-extra <synthetic>` does exactly this — the holdout is primary-only
  by construction, because mixing an easier corpus into the *holdout* would
  move the gate for a reason unrelated to the net.
* **Synthetic rows are also ~15-20× cheaper to harvest** (10–16 rows/s vs
  ~0.7 rows/s realistic, which drives full runs sequentially). They are both
  the cheaper data and the data that raises realistic-holdout performance.
* **The transformer is an improvement, not a prerequisite.** With enough
  synthetic the shipped MLP clears the gate at 0.462. The transformer is still
  better — 29% lower HP MAE — so prefer it if you are building fresh, but the
  existing Rust inference path, JSON export and parity test are not blockers.

Caveat: "more synthetic" is simultaneously more volume and more diversity, and
these runs cannot separate them (realistic data caps out at 14,998 rows, so a
volume-matched realistic control is not constructible from this harvest). The
recipe is unaffected either way.

### 2.1.1 Realistic harvests saturate, and that is the hard part

The realistic harvest came back at mean `win_rate` **0.956**, with 17,043 of
18,603 rows at exactly 1.0 and only 1,560 containing a death — i.e. the M1
confound is not a fluke of that run, it is what realistic self-play states at
**ascension 0** look like. The heuristic driver mostly enters fights it wins.

Two consequences:

* `--ascensions` now exists (it did not; realistic mode was pinned to A0 and
  also silently ignored `--character`). **But difficulty is NOT the cause of
  the saturation — hypothesis tested and rejected:**

  | asc | rows | mean win_rate | frac at exactly 1.0 |
  |---|---|---|---|
  | 0 | 75 | 0.929 | 0.893 |
  | 2 | 61 | 0.951 | 0.885 |
  | 5 | 68 | 0.926 | 0.882 |
  | 8 | 62 | 0.935 | 0.871 |
  | 10 | 53 | 0.874 | 0.849 |

  A0 → A10 moves the win rate only 0.929 → 0.874 and still leaves 85% of
  fights won outright. **Survivorship is the cause**: a run doing badly dies
  and stops emitting rows, so observed states are ones where the deck was
  already good enough to reach them. The synthetic path escapes this only by
  deliberately building bad decks. The fix is §2.1.0's mixture, not difficulty.
* Realistic runs also yield far fewer cells (40, of which 22 clear 8 held-out
  rows) versus 305 synthetic, because real states cluster. Expect noisier rho.

On the threshold: 0.20 was set blind. Readings so far are 0.76 (synthetic) and
0.28 (realistic). Treat it as a floor for "not a lookup table", not a target.

**What this still does NOT establish:**

- **The labels come from a weak combat AI** (K=12 probes at 64 sims). Deck
  rankings under campaign-strength play may differ.
- Ranking ability is necessary, not sufficient: it says nothing about whether a
  policy trained against the surrogate transfers. §2.2 still applies in full.

### 2.2 Model exploitation

A run policy trained against the surrogate optimizes against its errors. Shared
across both arms, so a *relative* comparison degrades gracefully — but a
feature that merely helps exploit the surrogate reads as a win. **Re-run any
surrogate-screen positive at low MCTS sims before it justifies a campaign.**

### 2.3 Given the measured curve, prefer cheap MCTS

Surrogate combat is only ~2× cheaper than `--sims 32 --boss-sims 100`, and it
costs a per-character harvest plus both risks above. Reach for it when you want
a **full-campaign-shaped** run (150 iterations × 256 games) inside an hour, or
when the question is not deck-shaped. For an ordinary screen, low sims is the
better trade.

---

## 3. Encoder experiment features

Candidate encoder inputs live behind `STS2_PPO_FEATURES` (`ppo/schema.py`).
With the flag unset the schema is **byte-identical** to what every committed
champion was trained under — this is load-bearing, not tidiness: `N_CAT` is the
width of the exported ONNX `cat` input and is mirrored by the deployed Rust
encoder's own `N_CAT` (`sts2-rs/crates/sts2-runai/src/ppo/encoder.rs:51`), so
bumping it unconditionally would break champion serving and the wire-parity
gates the moment it landed.

`ppo.screen` sets the var per arm. `ppo.trainer` refuses to start when its
`--features` disagrees with the schema the interpreter imported, and records the
value into every checkpoint's saved cfg.

### Currently implemented: `reward_upgrade` (GH #545)

Card-reward options carry the upgrade flag the engine rolled for them
(`roll_for_upgrade`: `act_index * 0.25`, halved at A7+, so 25% in act 2 and 50%
in act 3 on every non-rare option). The encoder passed `False` into `cat[1]`
unconditionally, so a free upgraded card featurized as the base card on the
most-exercised surface in training — 17.89 decisions/run.

**This one appends no slot**, which makes it the opposite of `boss` in the one
way that matters. Every tensor is identical with the flag on or off, so nothing
downstream can detect which arm produced a champion, and "an older champion
drops the new column" does not apply. The env var IS the pairing mechanism:

* Training: `export STS2_PPO_FEATURES=reward_upgrade` **and** pass `--features
  reward_upgrade` (the trainer cross-checks them and refuses to start on a
  mismatch, then records the value into the checkpoint's cfg).
* Evaluating a gated champion through the Rust arm (`run_arena`, the workbench
  backend, the advisor exe): the same var, honoured by
  `sts2-runai::ppo::features`. Forget it and the champion is served an
  observation it never trained on, silently.
* **Promotion should graduate it** — flip the branch unconditional and delete
  the gate, as `boss` did — rather than leave the fleet depending on an
  environment variable being set correctly forever.

Do NOT enable it for a champion trained without it. Every champion to date
holds an unconditional anti-upgraded prior (`wb-27d55ac0`); handed the flag
mid-life it reads a free upgrade as a reason to skip the card.

`ppo.dump_parity_fixtures` run with the var set writes a small supplementary
parity fixture holding only rows that exercise the gated branch — act 2+, since
act 1's odds are exactly zero and an act-1 fixture would pin nothing.

### Graduated: `boss`

Now unconditional (`CAT_BOSS`, keyed on the encounter id) — kept here for the
shape of the argument, not as a live flag.

Act-boss identity on the state token, keyed to the boss's lead monster through
the `monster` registry.

**No engine change was needed.** The boss is already in every observation:
`RunManager.full_map()` appends a boss meta-node carrying `kind` (lead monster)
and `id` (encounter id), resolved from `RunState::pinned_act_boss`. The encoder
simply never read it — `mapfeat.compute` takes a boss node's room type and
nothing else. It is player-visible (the map widget shows this label), so
reading it does not breach `env.observe`'s clairvoyance discipline. Only the
*current* act's boss is in the obs; future acts' bosses are not, and must not be
added.

Because the field was already in the obs, **existing recorded corpora already
contain it** — `ppo.feature_probe --feature boss` runs against a corpus recorded
before the feature existed.

### Shipping a feature is a separate, larger job

A screen win does not make a feature deployable. Shipping means moving the Rust
encoder, the ONNX export and the parity fixtures together, or the train/deploy
asymmetry behind GH #302 recurs. Budget that separately.

---

## 4. Calibration — the screen is worthless until this is done

The champion playbook records three confident mis-calls, all made by judging a
run on something other than a paired gate. An uncalibrated screen is a fourth
waiting to happen.

**Before the screen decides anything, run it on changes whose campaign outcome
is already known.** Pre-register the criteria in a commit first (champion
playbook §2.3).

| replay | how to run it | known campaign result | expected screen delta (B − A) |
|---|---|---|---|
| the #302/#317 event fix | comment out the `materialize_event()` bind in `env.py::_advance` for arm A | +15.7 pp | large **positive** |
| `--lr-anneal-final` | `--arm-b-flag lr_anneal_final=1.0` | +6.0 pp (PR #340) | **negative ~6 pp** |
| `LeafKind` narrow vs wide | `STS2_RUNAI_LEAF` | ≤1.3 pp, within noise | null |

**Mind the sign.** The screen reports `delta = B − A`, and in the anneal replay
arm B is the *degraded* arm: `lr_anneal_final=1.0` means the final LR equals
the initial one, i.e. no anneal at all — the default-OFF config whose absence
cost ~6 pp. A positive delta there is a FAILED calibration, not a pass.

**Mind the power — it is the binding constraint on this table.** McNemar over
`n` paired seeds needs `|b − c| ≳ 1.96·sqrt(n_discordant)`; at the ~37%
discordant rate measured on the boss-id screen that is a minimum detectable
effect of roughly

| eval seeds | MDE |
|---|---|
| 512 | ~5.5 pp |
| 1024 | ~3.8 pp |
| 2048 | ~2.7 pp |

So **512 seeds cannot calibrate this table.** The anneal replay's 6 pp sits
barely above the MDE — about 50% power, a coin flip on whether a real effect
registers — and the `LeafKind` row's ≤1.3 pp null is unfalsifiable, because a
screen blind below 5.5 pp reports null there whether or not the effect is real.
Use **≥1024** for the anneal replay. A "null" from an underpowered panel is not
evidence of a null; it is absence of evidence, and filing it as the former is
the exact mistake §4 exists to prevent.

Eval is cheap next to training — 1024 seeds × 2 arms is ~10% of one arm's
collection cost — so buy the power.

**Reusing an arm across replays is legitimate and saves hours.** `run_arm`
skips training when `<out>/<arm>/main.pt` already exists, and every replay in
this table shares the same arm A (preset defaults, `features=''`, same
`--seed-base`). Copy the `a/` directory from a finished screen into the new
`--out` and only arm B trains. Copy **only** `a/` — never the top-level
`eval-a.jsonl`, or the eval skip-if-exists will silently reuse the old panel
and `report()` will pair on the intersection, quietly discarding the extra
seeds you paid for.

Report sign and rough magnitude for each, in every world you intend to screen
in (cheap-MCTS and, if its gate passed, surrogate). If the ordering does not
survive, tune the preset — the sims / games / iterations split is the free
parameter — and re-run. If it still does not, say so: the screen is then a
triage tool for information-style questions (`ppo.feature_probe`) only, and the
honest move is to write that down rather than ship a number nobody can trust.

**Run the A/A test too — but vary `--seed-base`, or it measures nothing.**
`ppo.trainer` is **deterministic** given `(config, seed_base)`:
`torch.manual_seed(0)` and `update_rng = default_rng(0)` are hardcoded, worker
action sampling is `default_rng(seed)` off the game seed, and `play_batch`
orders rows by submission index. Two identically-configured arms at the same
`--seed-base` therefore produce identical nets and a delta of exactly 0.00 —
which reads as a clean pass and is worthless. The only stochastic axis is which
seeds get trained on:

```bash
ppo.screen --name aa-noise --seed-base 810000000 \
           --arm-b-flag seed_base=820000000 --eval-seeds 1024
```

### 4.1 Measured: the screen is unusable below ~8 pp

Three arms, identical config, differing only in `--seed-base`, all evaluated on
the same 1024-seed panel (2026-08-12, Ironclad):

| pair | delta | p |
|---|---|---|
| 800M → 810M | +2.05 pp | 0.318 |
| 800M → 820M | −0.10 pp | 1.000 |
| 810M → 820M | −2.15 pp | 0.272 |

Pure-noise range **−2.15 … +2.05 pp, RMS 1.72 pp**. The pairing and the eval
panel are sound — no A/A delta is significant, so the screen does not
manufacture effects.

**But the anneal replay's signal for a known +6.0 pp campaign effect was
−2.64 pp, against a 2.15 pp noise floor: S/N ≈ 1.23.** The screen cannot
separate a real 6 pp campaign effect from panel noise, and the "correct sign"
that replay produced is worthless — two of three *pure-noise* deltas were also
negative.

Working threshold: a delta must clear ~2× the noise RMS (≈3.5 pp) to be
readable, which after the measured 44% attenuation means a **campaign effect of
roughly 8 pp or larger**. The +15.7 pp event fix would show. The +6.0 pp anneal
does not. Feature-sized effects of 1–4 pp are invisible.

**More eval seeds do not help — this is measured, not assumed.** McNemar is
not broken (it correctly returned p = 0.27 / 0.32 / 1.00 on the noise pairs);
the experiment is underpowered at the *training panel*, and evaluation cannot
touch that. Power can only come from more training per arm, or from averaging
over k seed_bases (noise falls as 1.72/√k, so 3× needs k≈9 ≈ 18 arms — usually
worse value than simply training longer).

**This is structural, not a tuning problem.** The screen trains on 10,240 runs
against a campaign's 38,400. That 3.75× reduction *is* the panel noise, and it
is the same reduction that buys the speed. Note what survives: §1.1's offline
`ppo.feature_probe` has no training panel at all, so none of this applies to
it. That tier delivers; this one currently does not.

---

## 5. Operational notes

- **Everything is skip-if-exists.** A killed screen keeps its finished arm;
  re-running reuses `<out>/<arm>/main.pt` and `<out>/eval-<arm>.jsonl`. Delete
  them to force a redo.
- **Each arm's stdout goes to `<out>/<arm>/train.log`**, not to the console —
  poll the log, not the process (champion playbook §4).
- **Watch combat fallbacks at low sims.** A starved search falls back more
  often; `diag_mcts_fallbacks.py` and the `resolution_log` are where that shows
  up. A screen whose combat is falling back constantly is not measuring the
  same thing as one whose combat is merely weak.
- **`--sims` / `--boss-sims` are inert under `--combat surrogate`.** There is no
  search to budget; stop quoting them in that mode.
- **The trainer banners every deviation from campaign settings** (`SCREEN:`
  lines) precisely because the playbook's recurring failure is a silent flag —
  `--lr-anneal-final`'s default-OFF cost ~6pp and nothing in the log
  distinguished it.

---

## 6. What must never happen

Nothing produced by this harness may be promoted. A champion is trained at the
campaign combat budget, on the full A0–A10 curve, and gated at 200 paired seeds
per level on a fresh panel against the currently deployed net, under a rule
committed before the sweep ran. `docs/ppo-champion-training-playbook.md` §2–3 is
unchanged and unrelaxed.
