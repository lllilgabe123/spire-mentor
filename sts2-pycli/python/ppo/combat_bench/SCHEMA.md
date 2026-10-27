# combat_bench: record schema (v1)

`ppo.combat_bench` measures the combat pilot against pro humans on the same fight entry states
(GH #830). It has three subcommands that share these records:

- `build` writes a **fight set**.
- `run` writes a **played file** for one agent spec on one set.
- `report` reads one set plus N played files.

## Fight set: `data/combat_bench/ironclad_<name>_<version>.jsonl`

There is one JSON object per line, and one line per human fight. This table lists every field.

*(additive)* **Versions.** `build --version` names the outputs. The v1 sets are frozen: `pro`,
`devpool`, and `dev` (the contested panel that `devpanel` cuts from the dev pool). `build` with
the default `--version v1` applies v1's rules and rebuilds `pro` and `devpool` byte for byte from
v1's index. A later version writes its own files beside them (`ironclad_pro_v2.jsonl`,
`ironclad_devpool_v2.jsonl`) and applies the rules v1 got wrong (MANIFEST `rules`):

- `casefold_players`: one player per `username.strip().casefold()`. v1 keyed players on the exact
  string and split a top-15 pro across two spellings, so 10 of their v0.107.1 runs were left
  out of the pro set and 5 sat in the dev pool (3 of their fights are on `dev_v1`). `player`
  hashes the spelling seen first, so a player with one spelling keeps v1's id.
- `clean_ranking`: selection runs with `modifiers` or BaseLib cards, and, from builds up to
  v0.107.1, a modded enchantment or an id the game does not define, do not count toward the
  ranking (`selection.ranking_runs_excluded`). The vendored id tables do not know what v0.108.0
  and later added, so those two checks skip such builds.
- `quit_needs_no_reward`: see `human.quit` and the `abandoned_after_win` flag.
- `deck_downgrade_owed`: see "Build rules", Deck.
- `in_rl2_eval`: the flag below.
- `screened_dropped`: a pro set leaves out every fight of an earlier version's dev panel
  (`--bench-dir`, `ironclad_dev_v*.jsonl`): an arm chosen on that panel was screened on them.

`--extra-top N` adds the top-N cohort's pro set, `ironclad_proN_<version>.jsonl`, built by the
same rule in the same run. Its fight rows are the same as the `--top` set's for the players both
hold. The dev pool still drops only the `--top` cohort's runs, so a wider set can share players,
runs and fights with it (MANIFEST `selection.extra_tops.<N>.devpool_overlap`). `build` refuses
to overwrite a set or MANIFEST already in `--out-dir` unless `--force` is given.

| field | type | meaning |
|---|---|---|
| `key` | str | `"<file>:<floor>"`. It is unique within the set and is the pairing key everywhere. |
| `file` | str | spire-codex run file name (`<hash>.run.gz`) |
| `floor`, `act` | int | Map floor (1-based, as `extract_run` uses it) and act (1–3). |
| `room_type` | str | `monster` / `elite` / `boss` |
| `encounter` | str | Encounter id, for example `WATERFALL_GIANT_BOSS`. |
| `character` | str | `"Ironclad"` |
| `ascension` | int | 10 |
| `player` | str | Stable id: `"p_" + sha256(username)[:10]`, or `"anon_" + sha256(file)[:10]` when the run has no username. From v2 on, the username is the player's first-seen spelling (`casefold_players`). Usernames are never written to the set, but the id is a pseudonym, not anonymity: it is an unsalted hash, and `file` names a public spire-codex run whose content holds the username (see "Privacy"). |
| `cohort` | str | `pro` / `other` |
| `run_won`, `run_abandoned` | bool | Whole-run outcome flags. |
| `deck` | list[str] | Same encoding as `pilot_human_decks`: `ID[+1][@ENCH:amt]`. Rebuilt per copy by the backward replay (see "Build rules"); a Goopy amount is an estimate (`goopy_amount`). |
| `relics` | list[str] | Relics held at fight entry. This includes relics later removed via `relics_removed`, and leaves out Toy Box wax relics that have already melted. |
| `potions` | list[str] | What the PILOT is given at fight start, after the belt logic (see "Build rules"). Each is a potion id. |
| `belt` | list[str] or null | Reconstructed belt at fight entry, oldest potion first, or null if it can't be reconstructed. |
| `potion_slots` | int | The human's belt size at fight entry: 3, minus 1 at A4+ (TightBelt), plus slot relics. `run` passes it to `build_scenario_state(potion_slots=)`, so start-of-combat grants such as Petrified Toad's rock find the human's slot count. |
| `belt_unused` | bool | The reconstructed belt held a potion that the pilot does not start with (`potions`, plus the engine toad's rock). |
| `monsters` | list[str] or null | The human's roster (pets and summons filtered), passed as the `monsters=` override. Null means the encounter's own roll is used: either that roll already is the human's roster (the override drops per-monster spawn arguments, so it is only passed where it changes the roster), or the fight is flagged `roster_unmatched`. |
| `hp_before`, `max_hp` | int | Entry HP and max HP. |
| `n_enchanted` | int | Count of enchanted deck cards. |
| `human` | object | Described in the next table. |
| `flags` | list[str] | Fidelity flags, from the closed set in the flags list below. |

The `human` object holds the human's outcome for that fight:

| field | type | meaning |
|---|---|---|
| `hp_after` | int | |
| `damage_taken` | int | C# `DamageTaken`: clamped unblocked damage, including self-damage. |
| `hp_healed` | int | |
| `max_hp_gained` | int | |
| `max_hp_lost` | int | |
| `turns` | int | |
| `died` | bool | The human died in this fight: the run is not abandoned and `hp_after == 0`. Never true for a quit. `damage_taken` can be below the HP lost (`death_hp_unlogged`). |
| `quit` | bool | The run was abandoned *during* this fight: the run is abandoned, this is its last room and it is combat, `hp_after == 0`, and `damage_taken < hp_before + hp_healed`. From v2 on, also: the floor logs no reward taken (a card or relic choice, or `gold_gained` that Maw Bank's 12 on entering the room and Hand of Greed do not explain). A run abandoned on the reward screen logs `current_hp` 0 for the floor of a fight the human won; that fight is flagged `abandoned_after_win`. `potion_choices` is no evidence, because in-combat generation logs picks too. A win abandoned before any reward was taken stays a quit. |

**`flags` (closed set):**

| flag | meaning |
|---|---|
| `enchanted` | The deck has at least one enchanted card. |
| `fresh_relic_counters` | The deck holds a relic whose counter or charges matter in combat (Pen Nib, Nunchaku, Happy Flower, Pendulum, Lizard Tail, Ember Tea, Bone Tea, Tea of Discourtesy, …). The rebuild starts every counter fresh. |
| `fur_coat` | The run holds Fur Coat; its marked fights are not reproducible. |
| `tinker_time` | The deck has a Mad Science / Tinker Time variant card. |
| `potion_overflow` | The human used more potions than the belt holds. |
| `in_rl2_train` | This fight start was in rl1's or rl2's RL start pool. Reproduce that pool from `rollout_learn rl`'s shuffle and `n_states`, using the args in `logs/2026-10-18-rollout-net-recipe/run_rl{1,2}.sh`. |
| `in_rl2_gen` | This fight was in the gen1 imitation data, which is hbank minus the holdout, through `rollout_learn gen`. |
| `in_rl2_eval` | *(additive, v2 on, pro sets only)* The fight's key is in one of the recipe's RL eval and screen sets (`boss800`, `wg_fights`, `elite800`, `hall800`), on which the deployed playout net was evaluated and its temperature chosen. Every dev-pool fight is one by construction, so dev sets do not carry it. |
| `event_combat` | The fight came from an event room (`rooms = [event, monster]`). |
| `roster_unmatched` | The human's `monster_ids` could not be mapped, so the encounter rolls its own roster. |
| `belt_capped` | The human's belt had more slots than the engine's maximum of 5 (`MAX_POTIONS`; Alchemical Coffer). The pilot gets 5 slots and at most 5 potions. |
| `potion_generator` | *(additive)* An in-combat potion generator is present: Entropic Brew among `potions`, Alchemize in the deck, or Delicate Frond held. The engine runs it for the pilot and rolls its own potions (into a 3-slot belt, where the human's A10 belt had 2), so the pilot's potion supply is not the human's. |
| `goopy_amount` | *(additive)* A Goopy card's amount is an estimate. Goopy gains 1 per play (`Goopy.AfterCardPlayed`) and its block bonus is `amount - 1`; the log only has the amount at the enchant and at the end. See "Build rules". |
| `deck_unmatched` | *(additive)* The backward deck replay could not match a card event at this floor or later, so the deck may be off. From v2 on this includes a gain or transform undo whose copy lacks the upgrade or enchantment its record shows. None on the v1 or v2 sets; v1's replay accepted that mismatch silently (see "Build rules", Deck). |
| `death_hp_unlogged` | *(additive)* The human died with `damage_taken < hp_before + hp_healed`: HP was lost through `CreatureCmd.Kill` (The Insatiable's Sandpit), which `DamageTaken` does not count. `report` scores every death as all HP lost, so `L_h` is unaffected. |
| `abandoned_after_win` | *(additive, v2 on)* The human won this fight and abandoned the run on its reward screen (see `human.quit`): `quit` and `died` are false, and `hp_after` is `hp_before - damage_taken + hp_healed`, since the logged value is the abandon's 0. |

## Build rules: `build`

`build` checks each rule below against the v0.107.1 C# decompilation. Its module docstring has the
details.

- **Runs.** A run is kept when it is a single-player standard Ironclad A10 v0.107.1 run. Abandoned
  runs are kept too, and their completed fights count. A run is dropped when it has any
  `modifiers`, any relic, card or potion id that is not in `localization_eng`, BaseLib
  `save_dict` cards, or an enchantment the game does not define.
- **Belt.** `belt` replays each floor as a multiset: the belt, plus picked `potion_choices`, minus
  `potion_used` and `potion_discarded`. Every gain is logged as a pick, including purchases,
  rewards and in-combat generation (`PotionCmd.TryToProcure`). The belt has 3 slots, minus 1 from
  A4 (TightBelt), plus 2 for Potion Belt, 1 for Phial Holster and 4 for Alchemical Coffer. The
  replay must end on the run file's final `potions` and `max_potion_slot_count`. If it doesn't,
  the run's belts are all null.
- **`potions`.** The pilot is given, in this order:
  1. The drunk potions that were on the entry belt.
  2. Every held Fairy in a Bottle. It is the only `PotionUsage.Automatic` potion, and it triggers
     on death.
  3. The drunk potions that were generated mid-fight, except:
     - AnyTime potions (`PotionUsage.AnyTime`: Blood Potion, Fruit Juice, Foul Potion, Entropic
       Brew). One that was not on the entry belt was picked on the fight's floor, most likely as
       its reward, and drunk after the fight.
     - all of them when an in-combat generator is present (Entropic Brew given, Alchemize in the
       deck, Delicate Frond held; flagged `potion_generator`). The engine runs that generator for
       the pilot, so the potions it made for the human would be counted twice.

  The list is capped at the belt's slot count. With Petrified Toad, one drunk Potion-Shaped Rock
  is left out, because the engine's toad procures it at combat start just as the game's does, and
  that rock takes a slot. The scenario's combat belt always has 3 slots, so where the human's
  full A10 belt left their toad no slot, the pilot still gets a rock. The MANIFEST counts these
  fights as `toad_extra_rock`.
- **`monsters`.** `rooms[].monster_ids` lists the encounter's slots in order. After them come
  creatures added later whose ids are new. Pets (`OSTY`, `BYRDPIP`, `PAELS_LEGION`) are always
  dropped. A summon (Wriggler, Tough Egg, Parafright, Gas Bomb, the Gremlin Merc's gremlins, Eye
  With Teeth, the Fabricator's bots) is dropped when its summoner is already on the roster.
  - If the roster equals the engine's own roll, `monsters` is null.
  - Otherwise the human's roster is passed when two checks hold. First, the encounter must rebuild
    with an identical `state_hash` when its own roster is passed back through the override.
    Second, every spawn-argument kind in the human's roster must be on the encounter's own roster.
  - If either check fails, the fight is flagged `roster_unmatched`.
- **Relics.** A relic in `relics_removed` at floor g is held for fights at floors up to g. That
  starts after its last `relic_choices` pick BEFORE g (a pick of the same id on g itself is the
  copy that stays, as at the Relic Trader; it is used only when there is no earlier pick), or at
  the start of the run if it was never picked (a starter relic). Toy Box melts its wax relics in
  relic order, one at the end of every third combat after it was obtained.
- **Deck.** `pilot_human_decks.DeckReplay` starts from the final deck, the only exact per-copy
  record, and undoes each floor's card events from the last floor down. Within a floor it undoes
  enchantments, upgrades, downgrades (Reflections downgrades then upgrades, possibly the same
  card), removals (the copy comes back as its record shows it, upgrade and enchantment
  included), transforms (the final card leaves, the original comes back as recorded), then
  gains. An upgrade is undone on the most recently added upgraded copy held at that floor, never
  on one that must stay upgraded because it was gained upgraded. A copy still held after its own
  gain floor was undone was gained without a log entry and is dropped there. On the v1 and v2
  sets every event matches and every run replays back to the starter deck plus Ascender's Bane.
  - *(v2 on)* A downgrade (Reflections) is undone on a plain copy that still owes the upgrade it
    was gained with, else on the oldest plain copy. v1 always took the oldest, which gave a copy
    that was never upgraded a free +1 when Reflections downgraded one gained upgraded:
    `aba9d32b429efa1f.run.gz` Rupture+1 on floors 17-35 (10 fights of `pro_v1`) and
    `3b690055e25a13dd.run.gz:20` Battle Trance+1 (`devpool_v1`). A gain undo whose copy lacks its
    record's upgrade or enchantment is noted, which catches that error; a copy with more than its
    record is common (cards the log shows gained plain that no later event upgrades) and is not
    noted.
  - Goopy is the only enchantment whose amount changes (+1 per play, and it adds Exhaust, so at
    most +1 per combat). The amount at a fight is interpolated linearly in combats between the
    enchant (the event's amount) and the copy's last fight (its recorded amount), clamped to what
    one play per combat allows, and flagged `goopy_amount`. A Reflections clone records the live
    amounts: in `280eed78ee500a3e.run.gz` the estimate for floor 46 is {9, 11, 11, 14} against
    the clone's exact {8, 11, 11, 14}.
  - The forward count rule of the original extraction made `hbank` and the
    `logs/2026-10-18-rollout-net-recipe/*.jsonl` sets. It under-counts upgrades: an upgrade of a
    copy gained later counted against the copies already held, removed and transformed copies lost
    their upgrade, and the +1 went to the oldest copy rather than the one the final deck shows.
    The MANIFEST's `decks` counts the v1 fights where the two rules differ.
- **Event combats.** Fights in `rooms = [event, monster]` are extracted and flagged
  `event_combat`. The engine has none of the `*_EVENT_ENCOUNTER` encounters, so every one of them
  is excluded as unbuildable.
- **Engine check.** Every written fight is built with `build_scenario_state` and entered with
  `debug_enter_combat`, at its rep-0 seed and with its `monsters` and `potions`. A fight that
  fails is left out and counted by reason. If only the override fails, the fight falls back to
  `monsters = null` and is flagged `roster_unmatched`.

## `data/combat_bench/MANIFEST.json`, `MANIFEST_<version>.json`

The manifest is written by `build`: `MANIFEST.json` for v1 (which `devpanel` also records the dev
panel in), `MANIFEST_<version>.json` for a later version. It holds:

- The sha256 and counts of each set: fights by room, act and act/room, runs, players, human deaths,
  quits, flags, fights per player, and the Kish effective number of players.
- The corpus scan counts, and the counts at each filter stage.
  - *(additive)* `version`.
  - *(additive)* `corpus.index_rows_named` and `corpus.v0.107.1_A10_runs_named`: the index rows
    with a username. Only those can be ranked or be pro runs. Every run file fetched from
    spire-codex after v1's scrape carries no `username`, whatever month the run was played in
    (June and July runs among them); the files v1 had are unchanged. So the corpus has grown since
    v1 without adding a named run, and re-downloading files will not bring names back.
- The selection rule, and the pro cohort's overlap with `pros25.json`.
  - *(additive)* `rules`: `"v1 (frozen)"`, or the list of rules a later version applies.
  - *(additive)* `selection.players_with_several_spellings` (index-wide) and
    `selection.cohort_players_with_several_spellings`: players that more than one username
    spelling joins into (`casefold_players`); `selection.ranking_runs_excluded`: selection runs
    left out of the ranking, by reason (`clean_ranking`). Both are 0 / empty under v1's rules.
  - *(additive)* `selection.devpool_overlap` for the `--top` pro set and
    `selection.extra_tops.<N>.devpool_overlap` for each `--extra-top N` set: per dev set (this
    build's dev pool, then every `ironclad_dev*_v*.jsonl` in `--bench-dir`), the `players`, `runs`
    and `fights` (keys) the pro set shares with it, `fights_of_shared_players` and
    `fights_in_shared_runs` (the pro set's fights there), and
    `kish_n_eff_players_without_shared`. Players are matched across spellings, since v1 gave each
    spelling its own id. A pro set used as a test set should drop every shared player, not just
    the shared fights.
  - *(additive)* `selection.extra_tops`: per `--extra-top N`, the set's name, its cohort's overlap
    with `pros25.json`, its eval runs and wins, its Kish effective number of players, and its
    `devpool_overlap`.
  - *(additive)* `screened`: the dev panels read from `--bench-dir` (keys per panel) and the
    fights each pro set left out because a panel holds them (`screened_dropped`).
- The unbuildable counts by reason, the roster statuses and the potion-rule counts
  (`generator`, and the drunk potions left out by the generator and AnyTime rules).
- *(additive)* `decks`: per set, how the replayed decks compare with the forward count rule
  (`same`, `more_upgrades`, `fewer_upgrades`, `upgrade_moved`, `other`).
- The `rollout_learn` pool reproduction; *(additive)* `rl2_pools.eval_sets` names the sets
  behind `in_rl2_eval`.
- `held_out_runs`: every run file that has a fight in any of the build's sets.
  `ppo.combat_bench.holdout` takes the union over every MANIFEST in `data/combat_bench`, and
  `rollout_learn gen` and `rl` drop those runs unless `--allow-bench-overlap` is given. What they
  drop or keep is recorded in the net's `meta.bench_holdout`, whose `manifest` lists the MANIFEST
  files read.

**Privacy.** No username is written to the repo; the username index (`players_all.jsonl`,
`players_all_<version>.jsonl`) and the pro-cohort mapping (`pro_cohort.json`,
`pro_cohort_<version>.json`) stay in `--private-dir`. That is not anonymity. `player` is an unsalted hash of the username, and
every row's `file` names a public spire-codex run whose content holds the username, so anyone with
the corpus can recover who each player is and which players form the pro cohort. A salted id
would not help while `file` is there, and `file` is the pairing key. Treat the sets as linkable
public-corpus data; `data/combat_bench` is outside the public export allowlist.

## Played file: written by `run`

- **Line 1** is the header: `{"header": {...}}`.

  | field | meaning |
  |---|---|
  | `schema` | `"combat_bench.played.v1"` |
  | `set_name`, `set_sha256` | which fight set was played |
  | `spec` | The agent spec, a dict. For MCTS it has `kind`, `sims`, `boss_sims`, `env` (dict), `overrides` and `rollout_net`. |
  | `src_hash`, `build_git`, `pgo` | `sts2_rs.src_hash()`, `sts2_rs.build_info()` if present, and whether the wheel is a PGO build. A PGO and a plain wheel of the same source share `src_hash`. |
  | `reps`, `rep_offset` | `rep_offset + reps <= 50`: `rollout_learn gen` seeds its training data with reps 50 and up (`--rep0` 50) and the RL start pool with rep 77. |
  | `caps` | `{max_rounds, max_steps, max_sims_per_fight}`. `max_sims_per_fight` defaults to `400 * max(sims, boss_sims)` for an MCTS spec (1.6M for 1000 / 4000) and to 1.6M for a factory. |
  | `python`, `host`, `argv` | |
  | `cpus` | *(additive)* `os.cpu_count()` of the host that started the file. |
  | `watchdog_s` | *(additive)* The wall-clock limit per cell; 0 means off. |
  | `block_size`, `order_seed` | *(additive)* The submission order: every cell is shuffled with `random.Random(order_seed)`, then `block = position // block_size`. |
  | `select` | *(additive)* `{room, limit, keys_file_sha256, keys, n_fights, fights_sha256}`, the subset of the set that was played. `keys` lists the selected keys in set order when `--keys-file` was given and is null otherwise (room and limit are re-applied to the set). `fights_sha256` hashes the selected keys joined by `\n`. |

  *(additive)* `spec` details. `search_dialect` takes one of three values:
  - `belief-draw` for the deployed honest search.
  - `belief-draw-clairvoyant` for `--clair`. That flag sets `STS2_RUNAI_HONEST=0` and prepends the
    `belief_dialect=1` override knob.
  - `legacy` for any other `STS2_RUNAI_HONEST` mode.

  The engine applies the deployed dialect only under BeliefDraw, by default
  (`MctsConfig::apply_mode_dialect`). That dialect is the RL playout net, the budget-keyed c_puct,
  draw widening and the EndTurn widening retune. The knob applies the same dialect under
  Clairvoyant, so `--clair` is the deployed pilot with perfect information. A bare
  `STS2_RUNAI_HONEST=0` gives the older clairvoyant search instead.

  **B0 moved for Ironclad on 2026-10-25 (GH #830, commit `a70aa8bec`).** B0, the arm with no
  `STS2_*` variable, is the deployed pilot: the BeliefDraw dialect with the character's rl2 playout
  net. For Ironclad fights the run agent now adds its own layer on top: the knobs
  `removed_counts_forms=1,endturn_widening_c=23.148,draw_widening_c=4.0` in every fight, and the
  built-in BC move prior (PUCT, mix 0.5, T 1) in elite and boss fights. The other four characters
  are unchanged. The layer also applies under `--clair` (the `belief_dialect=1` knob turns it on),
  but not under a bare `STS2_RUNAI_HONEST=0`.
  - A wheel carries the layer iff `sts2_rs.builtin_prior_net_json("Ironclad")` returns text.
    The header's `src_hash` tells a pre-ship B0 file from a post-ship one, because a resume
    refuses a different `src_hash`.
  - To reproduce the pre-ship B0 on a new wheel, pass both off switches:
    `--overrides removed_counts_forms=0,endturn_widening_c=5.787,draw_widening_c=2.0 --env
    STS2_RUNAI_PRIOR_NET=off`.
  - `dev_v1` stays the frozen panel: it was selected by the pre-ship pilot on reps 40-47. Do not
    re-run `devpanel.py` against the new default.
  - Old `b0*` rows are not the deployed Ironclad pilot any more. Re-anchor B0 on the ship wheel
    before you measure a new lever delta against it.
  - `values_many` labels for Ironclad states (the oracle, deepbets) now score with forms on,
    because `deployed_value_config` follows the state's character.

  `env` is the worker environment in full: the
  runner scrubs every inherited engine seam (`STS2_RUNAI_*`, `STS2_ROLLOUT_*`,
  `STS2_TRACE_*`, `STS2_COMBAT_LOG`) and applies `env` before the first
  search. `--overrides`, `--rollout-net`, `--rollout-temp` and `--clair` become
  `STS2_RUNAI_MCTS_OVERRIDES`, `STS2_ROLLOUT_NET` (an absolute path),
  `STS2_ROLLOUT_TEMP`, and `STS2_RUNAI_HONEST=0` plus the `belief_dialect=1` knob, respectively. `rollout_net_sha256` hashes the
  net file. A factory agent has `kind: "factory"`, `factory`
  (`"pkg.module:fn"`), `args`, and the same env fields.

  A resume requires every header field to be identical except `argv`, `host`, `cpus` and
  `watchdog_s` (wall-clock only, so a retry may raise it).

- **Every other line** is one CELL, a (fight, rep) pair: `{"key": ..., "block": <int>, "rep": <rep>}`.
  - Cells are submitted in seeded-random order, in blocks of a fixed size, and `block` is the cell's
    block index. Reading only completed blocks keeps interim reads unbiased.
  - Resume skips cells whose (key, rep) is already present, after checking that the header matches.
    With `--retry-ends watchdog,error` it replays the cells whose record ended that way and appends
    a fresh record. A reader keeps the LAST record of a cell whose earlier record is `watchdog` or
    `error`, and the first one otherwise.
  - On a harness failure that is not a fight outcome, `rep.end == "error"` and `rep.error` holds a
    repr.
  - Before writing anything the runner plays one real decision of the first pending cell, and it
    refuses a spec that fails there (exit 2). A first block whose cells all end in `error` aborts
    the run (exit 2).

A `<rep>` object has these fields:

| field | type | meaning |
|---|---|---|
| `rep` | int | `rep_offset + i` |
| `seed` | int | `_fight_seed(key) * 100 + rep` (asserted `rep < 50`) |
| `end` | str | `won` / `died` / `round_cap` / `step_cap` / `sims_cap` / `no_action` / `watchdog` / `error`. Only `won` and `died` are scored. A `watchdog` end depends on the machine, so the report flags it loudly. |
| `died` | bool | `end == "died"` |
| `hp_end_combat` | int | Player HP at the moment combat ends, before any post-combat heal (0 if died). |
| `hp_after` | int | Run HP after post-combat effects (kept for continuity). |
| `max_hp_after` | int | |
| `dmg` | int | Raw sum of `unblocked` on player damage rows. Legacy; it overcounts overkill. |
| `dmg_c` | int | Sum of `hp_before - hp_after` on player damage rows. This is the C# `DamageTaken` definition: clamped, and it includes self-damage. |
| `turns` | int | Max combat-log round. |
| `steps` | int | Actions taken. |
| `decisions` | int | Steps where the agent searched (non-forced). |
| `sims_spent` | int | From `MctsCombatAgent.combat_budget_spent()[3]`; null for non-MCTS agents. |
| `cpu_s` | float | `time.process_time()` spent in the fight. |
| `potions_left`, `drunk` | int | As in `pilot_human_decks`. |
| `log_dropped` | int | Present only if the combat log hit its cap. |
| `tree_reuse` | object | Present only when the agent ran with `tree_reuse=1` (GH #830): `MctsCombatAgent.tree_reuse_stats()` for this fight — `searched`, `reused`, `reused_via_chance`, `fresh_*` by reason, `retained_visits`, `new_sims`, `forced_carried`, `forced_dropped`. Reuse rate `reused / searched`; carried-over share of root visits `retained_visits / (retained_visits + new_sims)`. |
| `root_filter` | object | Present only when the agent ran with `STS2_RUNAI_TURNPLAN_ROOT_FILTER=<weights>:<K>` (GH #830 turn-plan root filter, `docs/superpowers/specs/2026-10-22-turnplan-pro-root-filter-screen-design.md`). Counts over the cell: `decisions` (combat decisions the agent reported a filter record for), `applied` (root restricted), `fallback_forced` / `fallback_cap` / `fallback_empty` (root left unfiltered: one legal action, plan cap hit, no surviving plan starts with a legal action), `chosen_topk` / `chosen_anchor_only` (the pick starts a top-K plan / only anchor plans), `legal_n` and `allowed_n` (sums over applied decisions of the canonical legal root actions and of the actions left), `filter_ms` (enumerate + score + anchors) and `decide_ms` (whole `agent.step`), and `rows`: one list per decision, `[applied 0/1, fallback "" / "f" / "c" / "e", legal_n, plans_n, kept_n, allowed_n, chosen_via "" / "t" / "a", filter_ms, decide_ms]`. |
| `dmg_unlogged` | int | *(additive)* HP the player lost that no damage row records. The runner finds it by walking the exact HP at every step boundary together with the damage rows, in time order. The engine writes all player `Effect::DealHpLoss` without a row (`hooks.rs` `apply_dealhploss`): Thorns, Inferno, Combust, Crimson Mantle. C# counts these in `DamageTaken`, so the C#-equivalent damage is `dmg_c + dmg_unlogged`. It is a lower bound if a heal and an unlogged loss fall between the same two checkpoints. |
| `hp_end_inferred` | bool | *(additive)* Present only when true. The engine applies the victory heals (Burning Blood, Meat on the Bone, Black Blood) inside the combat state and logs no heal rows, so `hp_end_combat` comes from inverting them. This flag means HP moved after the last exact checkpoint with no row, and the end HP is the nearest HP consistent with the post-heal HP. |
| `hp_end_ambiguous` | bool | *(additive)* Present only when true. The inferred end HP was one of several consistent ones (Meat on the Bone makes the heal map non-monotonic), so it was not booked into `dmg_unlogged`. A heal capped at max HP can also hide an unlogged loss on the killing step; nothing flags that. |
| `load1` | float or null | *(additive)* The host's 1-minute load average when the cell started. |
| `host` | str | *(additive)* The host the cell ran on (a resume may run elsewhere than the header's `host`). |

*(additive)* An unscored end (a cap, `no_action` or `watchdog`) leaves combat
unfinished. Its `hp_end_combat`, `hp_after` and `max_hp_after` are the
in-combat values at the stop. An `error` record carries only `rep`, `seed`,
`end`, `died`, `error`, `cpu_s`, `load1` and `host`.

## Metrics: implemented in `report`

- **Per-rep HP-equivalent loss:** `L = min(dmg_c + dmg_unlogged, hp_before)` for a win and
  `L = hp_before + 65` for a death. `dmg_c + dmg_unlogged` is the pilot's C# `DamageTaken`
  equivalent: the game routes Thorns, Inferno, Combust and Crimson Mantle through
  `CreatureCmd.Damage` (counted at `CreatureCmd.cs:334`), but the engine logs no row for them. A
  rep without `dmg_unlogged` counts it as 0. The report's `dmg` metric is this same sum. A death
  counts all HP whatever the rows say: HP lost through `CreatureCmd.Kill` is in no
  `DamageTaken`.
- **Human:** `L_h` is the same with `damage_taken`.
- **Quit bracket:** a quit's counterfactual `L_h` lies in `[min(damage_taken, hp_before),
  hp_before + 65]`. The report gives every pilot-vs-human number at both ends,
  `quit_as_survived` (`died = 0`) and `quit_as_death` (`died = 1`), and with the quit fights
  dropped, `quit_excluded`. That is a missing-at-random point estimate, not an end of the
  bracket; the strata, player and encounter tables use it.
- **Fight level:** a fight's score is the mean of `L` over its scored reps. Arm-vs-arm Δ is paired on fights. Only fights with at least one scored rep in **both** arms count, and each arm keeps the intersection of its scored reps by rep id.
- **Uncertainty:** every pilot-vs-human CI, the splits table's included, is the wild cluster
  bootstrap by player; CR1 by run ignores the player effect and under-covers (~80-89% on the pro
  split layouts). The death CIs cover ~93% rather than 95% (rare deaths over ~12 unbalanced
  players), so one that only just excludes 0 is not decisive.
- **Interim reads:** a block with fewer cells than planned gets a `!!! PARTIAL BLOCK` warning;
  `--complete-blocks-only` drops such blocks.
- **Cost:** `sims_spent` is the primary cost metric. `cpu x base` is n/a when the arms ran on
  different hosts or wheel kinds (`pgo`), and a load difference gets a warning. `sims_cap` ends
  get a `!!! SIMS CAP` warning.
- **Death weight:** 65 is the default of `report --death-hp`.

What `report` expects of the played header, beyond the table above:

- `set_sha256` is the sha256 of the set file's raw bytes. The report hashes the `--set` file and
  refuses a played file whose `set_sha256` differs (`--allow-set-mismatch` makes it a warning).
- Played files whose `src_hash` differ are refused (`--allow-engine-mismatch` makes it a warning).
- `block_size` (int, optional) is the number of cells per block. `--complete-blocks-only` uses it.
  When it is absent, the report takes the largest per-block cell count instead, which is right once
  any block has completed.
- `reps` and `rep_offset` define the expected cells: every PLANNED fight times these rep ids. The
  planned fights are rebuilt from `select` (its `keys`, then `room`, then `limit`, in set order)
  and must hash to `select.fights_sha256`; a header without `select` plans the whole set. An
  expected cell that is missing or not `won`/`died` counts as dropped. The report covers the
  union of the arms' planned fights; arms that planned different fights get a warning, and
  their paired deltas use the common fights.

`report --json OUT` writes a `combat_bench.report.v1` object. Its top-level keys are `set`,
`death_hp`, `boot`, `perm`, `seed`, `baseline`, `arms` (name, path, header), `warnings`, `cells`
(end counts, missing, dropped, retried), `drop_bounds`, `human_gap` (per arm, then per quit
variant `quit_excluded` / `quit_as_survived` / `quit_as_death`: `overall`, and for
`quit_excluded` also `strata`, `players`, `encounters`), `paired` (per non-baseline arm:
`metrics`, `delta_bound_L`), `pro_gap_via_baseline`, `cost` (with `hosts`, `pgo`,
`load_per_cpu`), `cost_cpu_not_comparable`, `splits` (per arm: mean, `se_cr1`, wild `ci_lo` /
`ci_hi`, and the same for `deaths_pp`), and `self_null` when it was requested.
Non-finite numbers are written as null.
