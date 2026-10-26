# Release win rates — the five champions, A0–A10

**Status: done (2026-10-13).** 14,000 runs, 55 cells, 0 errored runs.
These are the numbers to quote for the release. The per-level tables in the
champion provenance files are promotion-gate measurements and are not
comparable to these or to each other (see "Why not the gate numbers").

> **Measured before the Ironclad search default (2026-10-25, GH #830, commit
> `a70aa8bec`).** The run agent now plays Ironclad with the forms + combo
> knobs in every fight and a built-in BC move prior in elite and boss fights.
> The Ironclad column below predates that change (and the `a10_it700`
> champion) and was not re-measured. To reproduce it on a newer wheel, set
> `STS2_RUNAI_MCTS_OVERRIDES=removed_counts_forms=0,endturn_widening_c=5.787,draw_widening_c=2.0`
> and `STS2_RUNAI_PRIOR_NET=off`. The other four columns are unaffected.

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
  (22.0% vs 11.0%, intervals disjoint).
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
