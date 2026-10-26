# Release win rates — the five champions, A0–A10

**Status: done (2026-10-13).** 14,000 runs, 55 cells, 0 errored runs.
These are the numbers to quote for the release. The per-level tables in the
champion provenance files are promotion-gate measurements and are not
comparable to these or to each other (see "Why not the gate numbers").

> **The Ironclad column was re-measured on 2026-10-26**
> (`2026-10-26-ironclad-release-winrates.md`), with the current champion
> `a10_it700` and the shipped Ironclad search default (GH #830: the forms +
> combo knobs in every fight and a built-in BC move prior in elite and boss
> fights), on engine `0181fa552`. Same protocol (`net_greedy`, 8,000 sims
> per decision in one tree, boss fights included), seeds and n: A0 96.0,
> A1 99.0, A2 93.5, A3 94.0, A4 95.0, A5 91.5, A6 93.5, A7 88.5, A8 84.5
> (n = 200 each), A9 56.4, A10 27.2 (n = 500 each), mean A0–A10 83.6.
> The tables below stay as the record of this sweep: their Ironclad column is
> `plr_it550` on engine `0cab615a3`. A newer wheel is not expected to
> reproduce it (untested): besides the old knob values and
> `STS2_RUNAI_PRIOR_NET=off`, it would need `rollout_policy=heuristic` to turn
> off the `rl2` playout net (PR #816, compiled in since 2026-10-18), and 690
> engine commits separate `0cab615a3` from `0181fa552`.
> The other four columns were not re-run. They are the latest measured
> release numbers for those characters, but not their current stack: each
> one's combat search has shipped a built-in playout net since 2026-10-21
> (GH #822–#825), after this sweep. The one release-protocol check since, a
> private one on 2026-10-21 (`defect-playout-net.md`), measured Defect A10
> with its net at 23.2% (116/500; `deckmut_it150` re-exported for the v0.111
> vocabulary, `net_greedy`, 8,000 sims, the same 500 seeds) against 22.0%
> below.

Data, driver and summary script: `logs/2026-10-11-release-winrates/`
(`summary.csv` per cell, `runs.csv` per run, `run.sh`, `summarize.py`). That
directory stays in the private monorepo; the public snapshot carries this
write-up, whose tables hold every per-cell number.

## Setup

| | |
|---|---|
| Models | the five champions in `sts2-pycli/data/ppo/champions.toml` at `0cab615a3`: Ironclad `plr_it550`, Silent `cont_it300`, Defect / Regent / Necrobinder `deckmut_it150` |
| Engine | `0cab615a3` (main to `c1dda6ad1` differs only by the SimBridge → SpireMentor rename) |
| Driver | `ppo.probe --policy net_greedy --act1 roll` |
| Combat search | MCTS, **8,000 sims per decision, boss fights included**, one search tree per decision (no root-parallel split) |
| Act 1 | rolled per seed: 7,515 Underdocks / 6,485 Overgrowth runs |
| Seeds | 81,000,000 onward (fresh panel, disjoint from every gate panel); the same seeds for every character |
| Size | 200 seeds per cell at A0–A8, 500 at A9–A10 |
| Wall clock | ~54 h on 32 cores |

The search budget is the advisor's (`realgame_advisor.py --sims 8000`), but
run as one tree. The advisor splits it across up to 8 root-parallel trees of
1,000; a single 8,000-sim tree is at least as strong, so these numbers may sit
slightly above what the live advisor plays.

## Results

Win rate in %, Wilson 95% interval in brackets.

| | Ironclad | Silent | Defect | Regent | Necrobinder |
|---|---|---|---|---|---|
| A0 | 89.5 [84.5–93.0] | 78.5 [72.3–83.6] | 93.0 [88.6–95.8] | 67.5 [60.7–73.6] | 66.5 [59.7–72.7] |
| A1 | 95.0 [91.0–97.3] | 80.0 [73.9–85.0] | 90.0 [85.1–93.4] | 77.5 [71.2–82.7] | 73.0 [66.5–78.7] |
| A2 | 92.5 [88.0–95.4] | 77.0 [70.7–82.3] | 92.0 [87.4–95.0] | 73.0 [66.5–78.7] | 73.0 [66.5–78.7] |
| A3 | 90.0 [85.1–93.4] | 74.0 [67.5–79.6] | 89.0 [83.9–92.6] | 72.0 [65.4–77.8] | 68.5 [61.8–74.5] |
| A4 | 86.5 [81.1–90.6] | 67.5 [60.7–73.6] | 90.5 [85.6–93.8] | 66.0 [59.2–72.2] | 62.5 [55.6–68.9] |
| A5 | 85.0 [79.4–89.3] | 64.0 [57.1–70.3] | 85.0 [79.4–89.3] | 59.5 [52.6–66.1] | 59.5 [52.6–66.1] |
| A6 | 86.0 [80.5–90.1] | 65.5 [58.7–71.7] | 87.5 [82.2–91.4] | 60.0 [53.1–66.5] | 52.5 [45.6–59.3] |
| A7 | 79.0 [72.8–84.1] | 68.0 [61.2–74.1] | 79.0 [72.8–84.1] | 51.0 [44.1–57.8] | 49.0 [42.2–55.9] |
| A8 | 70.0 [63.3–75.9] | 55.5 [48.6–62.2] | 70.0 [63.3–75.9] | 43.5 [36.8–50.4] | 44.5 [37.8–51.4] |
| A9 | 39.0 [34.8–43.3] | 20.0 [16.7–23.7] | 39.2 [35.0–43.5] | 15.6 [12.7–19.0] | 13.8 [11.1–17.1] |
| A10 | 11.0 [8.5–14.0] | 9.0 [6.8–11.8] | 22.0 [18.6–25.8] | 6.2 [4.4–8.7] | 5.6 [3.9–8.0] |
| **mean A0–A10** | **74.9** | **59.9** | **76.1** | **53.8** | **51.7** |

The mean is unweighted across the 11 levels.

- Defect and Ironclad are tied through A9; at A10 Defect wins twice as often
  (22.0% vs 11.0%, intervals disjoint). This holds for the 2026-10-11
  Ironclad column only. Re-measured on 2026-10-26 (`a10_it700`, the #830
  default, engine `0181fa552`), Ironclad is ahead of this Defect column at
  A9 (56.4% vs 39.2%, n = 500 each, intervals disjoint, two-proportion
  p ≈ 5e-8). At A10 the gap (27.2% vs 22.0%) is within noise: the intervals
  overlap ([23.5–31.3] vs [18.6–25.8]), p ≈ 0.06. Against Defect with its
  own playout net (23.2% at A10, same seeds and budget; note above) the gap
  is 4.0pp, p ≈ 0.15. These comparisons cross engines and search stacks, so
  they are descriptive.
- Silent is third. Regent and Necrobinder are statistically tied from A3 on.
- Every character drops 15–30pp from A8 to A9 and roughly halves again at
  A10. The old gate panels show the same A8 → A9 cliff, so it is a property of
  those ascensions, not of any one net.
- A1 above A0 for several characters is inside the noise. Do not read A1 as
  easier.

### Act 1 split

| | Overgrowth (n=1,297) | Underdocks (n=1,503) |
|---|---|---|
| Ironclad | 63.2 | 65.0 |
| Silent | 46.3 | 53.6 |
| Defect | 66.2 | 66.5 |
| Regent | 43.5 | 45.6 |
| Necrobinder | 39.8 | 45.2 |

Pooled over A0–A10, per character. The act is a function of the seed, so this
is a by-seed split, not a paired comparison. Even so, no character does
worse on Underdocks, including Defect, Regent and Necrobinder, which never
trained on it.

## Why not the gate numbers

The champion provenance files carry per-level tables, but each one comes from
its own seed panel, its own engine build, and the cheap 1,000 / 4,000 sim
budget, and three of them ran Overgrowth only. At A0, A3, A6 and A10 this panel
reads up to +22pp above them (Necrobinder A6: 30.0 → 52.5; Defect A10:
11.5 → 22.0), with Regent and Ironclad A0 the exceptions. The search budget is
the likely driver, but this panel changes budget, Act 1 mix and seeds at once
and attributes nothing among them.

## Reproduce

```bash
logs/2026-10-11-release-winrates/run.sh      # resumes; ASC_ORDER reorders levels
cd logs/2026-10-11-release-winrates && python3 summarize.py
```
