import type { IntentModifier } from "./intentBreakdown";
// Every /api/* call goes through apiFetch: a plain fetch against the Python
// backend, or — in the browser-only build (VITE_WORKBENCH_MODE=static) — the
// WASM engine in a Web Worker. See src/transport/.
import { apiFetch, configuredAssetsBase, isStaticMode } from "./transport";
import { bridgeBase, bridgeFetch } from "./bridge";
import { gameVersionBody } from "./gameVersion";
// Static mode has no server to build a bug report: the page builds the dump
// and posts it to the host's Pages Function. See src/bugreport.ts.
import { captureBrowserBugReport, fetchBrowserBugReport } from "./bugreport";

export interface Card {
  index: number;
  id: string;
  name: string;
  cost: number;
  type: string;
  rarity?: string;
  // Frame tint key for the card skin — the card's POOL colour, not the run
  // character's ("colorless" for a Colorless card in an Ironclad deck). Set by
  // the backend, which is the only side that knows the card's own character.
  color?: string;
  description: string;
  // The same text with the game's own `[gold]…[/gold]` keyword markup left in.
  // Only the card skin uses it; everything else reads `description`.
  description_rich?: string;
  // Regent star cost — a second cost resource alongside energy, drawn as its
  // own gem. Only the 21 cards with a CanonicalStarCost carry it.
  star_cost?: number | string;
  // The card's PRINTED energy cost, when the backend knows it. `cost` above is
  // the EFFECTIVE one, so the pair is what says a cost was modified: the frame
  // paints the numeral chartreuse below the base and energy-blue above it,
  // like `CardCostHelper.GetEnergyCostColor`. Combat cards only.
  base_cost?: number | null;
  upgraded?: boolean;
  stats?: Record<string, number>;
  // Per-enemy fully-modified per-hit attack damage (Strength/Vulnerable/Weak/
  // relics folded in), keyed by the enemy's full encounter-array index as a
  // string. Combat hand cards only; absent for non-attacks and older backends.
  damage_preview?: Record<string, number>;
  // The block this card would really grant right now, Dexterity / Frail /
  // Fasten folded in. Combat hand cards only, and only when it differs from the
  // printed `stats.block` — a Defend under Dexterity 5 prints 5 and grants 10,
  // and showing the printed number made Necrobinder's summon cards look like
  // the better mitigation (GH #520).
  block_preview?: number;
  can_play?: boolean;
  /** WHY `can_play` is false, when the backend knows: `"cost"` (short on
   *  energy) or `"blocked"` (a hook/affliction forbids it — a Bound card after
   *  another Bound card was played this turn). The game paints the two
   *  differently, so the frame needs the distinction. Absent on backends and
   *  recorded logs that don't ship it. */
  unplayable_reason?: string;
  /** The card's affliction, as the game names it ("Bound", "Galvanized"). The
   *  server appends its extra card text to the description, so the frame gets
   *  the words for free; this is the flag itself, for anything that needs the
   *  state rather than the text. */
  affliction?: string;
  affliction_amount?: number;
  // Upgrade selects (rest-site Smith, Armaments, …) ship the upgraded twin of
  // every un-upgraded option so the pick renders as a before → after
  // comparison, like the game's own upgrade preview. Absent everywhere else.
  upgrade_preview?: Card;
  // Keywords the upgrade adds/removes ("+Innate"), on the twin above — the only
  // visible change for cards that upgrade by keyword alone.
  keyword_delta?: string;
  image: string | null;
  target_type?: string;
  card_cost?: number;   // shop: energy cost (the shop's `cost` field is the gold price)
  on_sale?: boolean;
  /** Buyable RIGHT NOW: in stock AND affordable. Gates the buy action. */
  is_stocked?: boolean;
  /** The slot still holds an item — false only once it has been BOUGHT. This
   *  is what "Sold" means; being unable to afford something is not that, and
   *  reading `is_stocked` as availability stamped "Sold" over half the shop. */
  is_available?: boolean;
}
export interface Power {
  /** Display title ("Painful Stabs"), already localized by the backend. */
  name: string;
  /** Engine registry id ("PAINFUL_STABS", "DOOM"). Present on the Rust
   *  backend; match on this rather than on `name`, which is a display
   *  string and differs from the id by spacing and casing. */
  id?: string;
  amount: number;
  display_amount?: number;
  description?: string;
  /** The 64x64 tray sprite (`PowerModel.PackedIconPath`). What `Powers.tsx`
   *  renders at 26px in the status tray. */
  image?: string | null;
  /** The 256x256 `BigIcon` (`PowerModel.BigIconPath`) the power VFX draw —
   *  `NPowerAppliedVfx`, `NPowerRemovedVfx` and `NPowerFlashVfx` each assign it
   *  to their own sprite. Optional: a deployed bucket can still be serving an
   *  art tree extracted before that pass existed, in which case the panels and
   *  the flash fall back to `image` as they did before. */
  image_big?: string | null;
}
export interface Enemy {
  index: number;
  /** The engine's `CreatureUid`. `index` is a board slot; combat-log rows key
   *  their per-hit records by uid, and this is the join between the two.
   *  Two enemies of the same species share a display name but never a uid. */
  uid?: number;
  name: string;
  /** The raw `MonsterKind`, PascalCase — `server.py`'s `normalize_kind` folds
   *  both producers' spellings (the engine's `DecimillipedeSegmentFront`, the
   *  bridge's `DECIMILLIPEDE_SEGMENT_FRONT`) onto it. It is what disambiguates
   *  the segments that share one display name, and it is also the decomp file
   *  stem, which is what lets `deathFadeFor` join against the published table.
   *  Absent on a fixture, and on any producer that predates it. */
  kind?: string | null;
  hp: number;
  max_hp: number;
  block: number;
  intents?: {
    type: string;
    damage?: number;
    /** The telegraph BEFORE Strength/Weak/Vulnerable; omitted when it equals
     *  `damage`. `modifiers` names the listeners that produced the gap. */
    base_damage?: number;
    hits?: number;
    intent_image?: string | null;
    modifiers?: IntentModifier[];
  }[];
  powers?: Power[];
  /** Cards this enemy is holding hostage right now — the Thieving Hopper's
   *  SwipePower. The game pins the stolen card ON the thief; killing it before
   *  it escapes is what gets the card back, so WHICH card it took is the whole
   *  decision. Absent (not `[]`) for every enemy that has stolen nothing. */
  stolen_cards?: Card[];
  image: string | null;
  rig?: string | null;
  rig_scale?: number | null;
  rig_bounds_h?: number | null;
  rig_bounds_w?: number | null;
  /** Per-instance size and tint from `NCombatRoom.RandomizeEnemyScalesAndHues`:
   *  the game sizes every member of a group of two or more monsters of one type
   *  by where its rolled max HP landed in that monster's initial-HP band, and
   *  tints them apart. Present only where the rule fires — a lone monster or a
   *  fixed-HP one is drawn at its authored size, so absent means 1.0 / none.
   *  Applies to the DRAWN rig, never to the Bounds width the row spaces by. */
  size_jitter?: number | null;
  hue_jitter?: number | null;
  // Multi-part bosses (Kaiser Crab) share ONE rig: `rig_group` ties the members
  // together (same group = one skeleton rendered once), `rig_part` ("left"/
  // "right") is the animation-track family this arm drives.
  rig_group?: string | null;
  rig_part?: string | null;
  // Test Subject only: which of its 3 revive forms (more heads each time) is
  // currently alive, so the frontend can pick idle_loop{N}. Absent/1 = form 1.
  rig_form?: number | null;
  // Override idle clip for monsters whose resting pose depends on combat state
  // (Spiny Toad idle_naked_loop vs spiked idle_loop; Tunneler hidden_loop while
  // burrowed). Absent → the rig's default idle. The backend derives it from the
  // enemy's powers so the workbench matches the game's animator branches.
  rig_anim?: string | null;
}
export interface Relic { name: string; description?: string; index?: number; cost?: number; is_stocked?: boolean; is_available?: boolean; image?: string | null }
export interface Potion { index: number; name: string; description?: string; cost?: number; is_stocked?: boolean; is_available?: boolean; image?: string | null }
export interface Option {
  label: string;
  action: string;
  args: Record<string, unknown>;
  enabled: boolean;
}
export interface EventOption {
  index: number;
  title?: string;
  name?: string;
  description?: string;
  is_locked?: boolean;
  image?: string | null;
}
export interface RestOption {
  index: number;
  option_id?: string;
  name?: string;
  is_enabled?: boolean;
}
export interface Bundle { index: number; name?: string; cards?: Card[] }
export interface MapChoiceNode { col: number; row: number; type?: string | null; icon?: string | null }

/** Click handler shared by all interactive decision screens (Play mode).
 *  Real implementations (``liveAction`` in App.tsx) are async; the union lets
 *  callers that need to sequence follow-up calls (e.g. CardSelect's
 *  accumulate-then-finalize) ``await`` it while fire-and-forget callers and
 *  test stubs can still pass a plain sync no-op. The result type is loose
 *  because screens ignore it, while ``liveAction`` resolves to the Step it
 *  landed on so a caller stepping through a multi-action decision can see
 *  where it got to. */
export type ChooseHandler = (
  action: string,
  args: Record<string, unknown>,
) => void | Promise<unknown>;

/** Necrobinder's Osty companion (a player-side minion). Absent for other characters. */
export interface Osty {
  name?: string;
  hp?: number;
  max_hp?: number;
  block?: number;
  alive?: boolean;
  image?: string | null;
  rig?: string | null;
  rig_scale?: number | null;
  rig_bounds_h?: number | null;
  rig_bounds_w?: number | null;
}

/** A Defect orb in the orb queue. Absent for other characters. */
export interface Orb {
  index: number;
  name: string;
  type: string;       // "Lightning" | "Frost" | "Dark" | "Plasma" | "Glass"
  passive: number;    // per-turn passive value
  evoke: number;      // value released when evoked
  image?: string | null;
}

export interface CombatState {
  decision: string;
  round: number;
  // The fight's log so far, on a REPLAYED or followed live-game step (the
  // workbench's own live sessions carry it per-node on the timeline instead —
  // see `activeNode`). Server-rendered; absent on older run logs.
  combat_log?: CombatLogRow[];
  energy: number;
  max_energy: number;
  draw_pile_count: number;
  discard_pile_count: number;
  exhaust_pile_count: number;
  // Full pile contents (combat only). The server scrambles draw_pile order so
  // the client can't peek at the real draw order; discard/exhaust are faithful.
  draw_pile?: Card[];
  discard_pile?: Card[];
  exhaust_pile?: Card[];
  player: {
    name: string;
    hp: number;
    max_hp: number;
    block: number;
    gold?: number;
    image?: string | null;
    rig?: string | null;   // live Spine rig base path (frontend appends .skel/.atlas/.png)
    rig_scale?: number | null;     // authored scene scale (preserves player↔enemy proportions)
    rig_bounds_h?: number | null;  // authored Bounds HEIGHT — the size hint for a rigless combatant
    rig_bounds_w?: number | null;  // authored Bounds WIDTH — what the row layout spaces by
    relics?: Relic[];
    potions?: Potion[];
    max_potion_slot_count?: number;
    // Full master deck (every decision). Browsable from any screen.
    deck?: Card[];
    deck_size?: number;
  };
  player_powers?: Power[];
  // Necrobinder only — present (with alive:false when slain) once Osty exists.
  osty?: Osty;
  // Defect only — the orb queue and its capacity (slot count). Present once the
  // Defect channels an orb; absent for other characters.
  orbs?: Orb[];
  orb_slots?: number;
  enemies: Enemy[];
  hand: Card[];
  energy_image?: string | null;
  // card_reward decisions
  cards?: Card[];
  can_skip?: boolean;
  gold_earned?: number;
  options?: Option[];     // normalized non-combat choices (action contract / fallback)
  victory?: boolean;      // game_over
  // shop
  relics?: Relic[];
  potions?: Potion[];
  card_removal_cost?: number;
  // event_choice / rest_site (their own option shapes; `options` above is the
  // normalized contract, these are the rich per-decision payloads)
  event_name?: string;
  description?: string;
  art?: { illustration?: string | null };
  event_options?: EventOption[];
  rest_options?: RestOption[];
  // card_select / bundle_select
  min_select?: number;
  max_select?: number;
  bundles?: Bundle[];
  // map_select (live)
  choices?: MapChoiceNode[];
  // Run progress, present on map_select / game_over states (carried forward by
  // the UI elsewhere). `floor` is the cumulative total floor across all acts.
  floor?: number | null;
  act?: number | null;
  ascension?: number | null;
  // What a room the engine resolved on its own just handed you. A treasure
  // chest is auto-opened on entry (see `_treasure_reveal` in the backend), so
  // this rides on the NEXT decision's state — usually the map — and is the
  // only thing that makes the room visible at all.
  room_reveal?: RoomReveal;
}

/** Display payload for a room resolved without a decision (a chest, today). */
export interface RoomReveal {
  kind: "treasure";
  act: number;
  gold: number;
  /** Null for an empty chest (Silver Crucible's veto). */
  relic: Relic | null;
}

/** One run-mode step's debug description (see runModeStep). */
export interface FaithfulDebug {
  phase: string;
  is_combat?: boolean;
  action?: string;       // resolved action label ("Play Strike", "Choose path", …)
  step?: number;         // run step counter after this action
  summary: string;       // one-line human-readable description for the log
  // combat decisions
  sims?: number;
  seed?: number;
  is_boss?: boolean;
  // card-reward decisions
  reward?: {
    encounter?: string;
    sims?: number;
    candidates?: { idx: number | null; label: string; score: number; win: boolean }[];
  };
}
/** The combat MCTS root distribution for a move the run-mode stepper already
 *  applied (a run-mode step's `search` field).
 *
 *  These rows describe the position the search ROOTED at — the one BEFORE
 *  `action` was played, not the `step` returned alongside them. That is why
 *  they arrive in their own field instead of `step.scores`: those are matched
 *  onto the rendered hand by `stable_idx`, and the hand has already shifted.
 *  See searchPanel.ts for how the UI keeps the distinction visible. */
export interface LastSearch {
  scores: Score[];
  /** Resolved label of the move this search chose ("Play Strike → Cultist"). */
  action: string;
  /** The board the search saw, so a row's `target_index` resolves to the name
   *  that slot held THEN. The live board is a move later — anything the move
   *  killed is already gone from it, which would shift every label. */
  enemies: Enemy[];
}
export interface Score {
  stable_idx: number;
  action: string;
  win_prob: number;
  q: number;
  visits: number;
  is_best: boolean;
  wins?: number;
  losses?: number;
  ties?: number;
  q_saturated: boolean;
  low_confidence: boolean;
  target_index: number | null;
}
/** Compact recorded-advice row (live-game logs): the advisor's pick, its
 *  one-line note, and the policy-ranked options for decisions the MCTS never
 *  searched (map / shop / rewards have no rollout visits). */
export interface StepAdvice {
  best?: string | null;
  note?: string | null;
  options?: { label: string; share?: number | null; is_best?: boolean }[];
}
export interface Step {
  idx: number;
  decision: string;
  state: CombatState;
  scores: Score[];
  pv: string[] | null;
  chosen_action: Record<string, unknown> | null;
  action_label?: string;
  source: string;
  /** `bg_seed`/`bg_row`/`bg_col` are the run's master RNG seed and the map node
   *  this fight stands on — `NCombatRoom.GenerateBackgroundRngForCurrentPoint`
   *  needs exactly those three to roll the room's layer variants the way the
   *  real fight did. Absent on a producer that does not know its node, which
   *  falls the room back to a stable per-floor hash. */
  meta: {
    turn: number | null; step_no: number; floor?: number | null; act?: number | null;
    act_name?: string | null; advice?: StepAdvice | null;
    /** GH #878: a decimal string when the seed is past 2^53 (v0.111.0's `ulong`). */
    bg_seed?: number | string | null; bg_row?: number | null; bg_col?: number | null;
    /** GH #801: the session's game build; absent = v0.107.1. */
    game_version?: string;
  };
  /** GH #801: the session's game build; absent = v0.107.1. */
  game_version?: string;
}
/**
 * Whether the loaded recording can be taken over interactively, and where.
 *
 * `"combat"` — a fixture-sourced combat recording: only its combat steps.
 * `"run"` — a workbench run log that rebuilds from its seed: ANY step.
 * `null` — it cannot, and `reason` says why (shown instead of a dead button).
 */
export interface ResumeCapability {
  kind: "combat" | "run" | null;
  reason: string | null;
}
/** The live coach's records, as the real-game advisor writes them into a run
 *  log (`type: "coach"` lines). Field names follow PROTOCOL.md, "`coach`"
 *  (v5.1); `GET /api/coach` serves them. */
export interface CoachTurnContext {
  hand: string[];
  energy: number;
  max_energy: number;
  hp: number;
  max_hp: number;
  block: number;
  enemies: { name: string; hp: number; intent: string }[];
}
export interface CoachTurnRecord {
  kind: "turn";
  id: string;
  node_id: string | number | null;
  turn: number | null;
  floor: number | null;
  /** `ungraded` appears only in logs: the mod is never sent one. */
  verdict: "best" | "tie" | "slip" | "blunder" | "ungraded";
  you: string[];
  ai: string[];
  /** What one line did on the board that the other did not ("Your line
   *  wakes Lagavulin Matriarch on turn 1"); absent / null when nothing. */
  why?: string | null;
  hp_drop: number | null;
  win_before: number | null;
  win_after: number | null;
  peeked: boolean;
  ungraded_actions: number;
  log_name?: string | null;
  log_idx?: number | null;
  severity?: number;
  context?: CoachTurnContext | null;
}
export interface CoachChoiceRecord {
  kind: "choice";
  id: string;
  node_id: string | number | null;
  decision: string;
  label_kind: string;
  verdict: "agree" | "tossup" | "flag" | "ungraded";
  strength?: string;
  played: { label: string; prob: number; rank: number };
  top: { label: string; prob: number };
  alternatives: { label: string; prob: number }[];
  peeked: boolean;
  path?: { you: unknown[]; ai: unknown[] };
  log_name?: string | null;
  log_idx?: number | null;
}
export interface CoachRecapRecord {
  kind: "recap";
  id: string;
  node_id: string | number | null;
  log_name: string | null;
  log_idx: number | null;
  floor: number | null;
  room_type: string | null;
  hp_start: number;
  hp_end: number;
  hp_lost: number;
  /** Reserved, always null today (GH #856). */
  ai_hp_lost: number | null;
  turns: number;
  turns_graded: number;
  turns_ungraded: number;
  unfinished_turns: number;
  costliest: CoachTurnRecord[];
}
export interface CoachAutopsyRecord {
  kind: "autopsy";
  id: string;
  log_name: string | null;
  floor: number | null;
  act: number | null;
  items: (
    | { kind: "turn"; record: CoachTurnRecord }
    | { kind: "choice"; record: CoachChoiceRecord }
  )[];
  turns_ungraded: number;
  decisions_graded: number;
}
export type CoachRecord = CoachTurnRecord | CoachChoiceRecord | CoachRecapRecord;
/** `GET /api/coach`: the loaded run's records, withdrawals applied.
 *  `by_step` maps a replay step index to the ids of the graded records on it. */
export interface CoachLog {
  records: CoachRecord[];
  by_step: Record<string, string[]>;
  autopsy: CoachAutopsyRecord | null;
}
export interface TimelineEntry {
  idx: number;
  decision: string;
  label: string;
  details: string[];
  // This step's action's combat-log display rows: recorded per action by the
  // workbench save path, or sliced server-side off bridge states' raw log.
  // Empty for runs recorded before either existed.
  combat_log?: CombatLogRow[];
  turn: number | null;
  floor: number | null;
  act: number | null;
  act_name: string | null;
}
export interface MapNode { col: number; row: number; type: string | null; chosen: boolean; icon?: string | null }
export interface MapRow {
  idx: number;
  floor: number | null;
  act: number | null;
  act_name: string | null;
  nodes: MapNode[];
}
export interface FullMapNode {
  col: number;
  row: number;
  type: string | null;
  children?: { col: number; row: number }[];
  visited?: boolean;
  current?: boolean;
  /** Real in-game room-type badge URL (server-resolved); null when art is missing. */
  icon?: string | null;
}
/** One candidate first move in an {@link PathPlan}, with its optionality
 *  flexibility stats (mirrors Rust ``PyRunManager::path_plan`` columns). */
export interface PathPlanColumn {
  col: number;
  score: number;
  baseline_score?: number | null;
  n_paths: number;
  min_elites: number;
  max_elites: number;
  can_skip_elite: boolean;
  is_pick: boolean;
}
/** Optionality-aware path plan overlay (run-AI pathing surface). */
export interface PathPlan {
  best_col: number;
  baseline_col: number;
  /** Optionality picked a different first move than the legacy hard-max. */
  diverged: boolean;
  /** Planned route to the boss, as ``(col,row)`` coords (boss excluded). */
  route: { col: number; row: number }[];
  columns: PathPlanColumn[];
  /** The first hop was SEEDED from the column the driving run policy picked
   *  (the PPO brain), rather than chosen by the pathing surface itself — so
   *  `route[0]` is the move that will actually happen and `best_col` is what
   *  the pathing surface would have done. Null/absent = unseeded. */
  first_col?: number | null;
}
/** The act boss (and, at A10, the second boss) as a meta-node: coords plus the
 *  encounter's identity, which the map needs to draw the right silhouette. */
export interface BossNode {
  col: number; row: number; type: string;
  id?: string; name?: string; icon?: string | null;
}
export interface FullMap {
  rows: FullMapNode[][];
  boss?: BossNode | null;
  /** A10 DoubleBoss: the second boss node sitting directly above the act boss.
   *  It is ALSO a normal `rows` entry (typed `Boss`) — this carries the
   *  identity of the DIFFERENT encounter fought there. Absent below A10. */
  second_boss?: BossNode | null;
  current_coord?: { col: number; row: number } | null;
  /** Optionality-aware path plan (Rust backend only; absent on C#/replay). */
  path_plan?: PathPlan | null;
}
export interface MapGraph {
  idx: number;
  act: number | null;
  act_name: string | null;
  floor: number | null;
  full_map: FullMap;
}

/** Deck-analysis report from /api/{step/{idx},live}/deck-analysis. */
export interface DeckAnalysis {
  empty?: boolean;
  deck_size: number;
  assumptions: { energy: number; hand: number; samples: number };
  scores: { damage: number; block: number; velocity: number; volume: number };
  damage: {
    per_turn: number;
    single_target: number;
    vs_group: number;
    scaling_cards: string[];
    aoe_attacks: string[];
    attack_count: number;
  };
  block: { per_turn: number; scaling_cards: string[] };
  velocity: {
    tags: Record<string, number>;
    engine_cards: number;
    cards_per_turn: number;
    extra_per_turn: number;
  };
  composition: {
    by_type: Record<string, number>;
    by_rarity: Record<string, number>;
    energy_curve: Record<string, number>;
    dead_cards: number;
    unupgraded_basics: number;
  };
  warnings: { severity: string; text: string }[];
}

type AxisScores = { damage: number; block: number; velocity: number; volume: number };

/** Per-candidate axis shift for the card-reward preview. */
export interface RewardPreview {
  base: AxisScores;
  candidates: { index: number | null; scores: AxisScores; delta: AxisScores }[];
}

/** One offered option's Monte-Carlo act outcome from the card-pick advisor. */
export interface AdvisorCandidate {
  idx: number | null;       // pick_card_reward option index (null = skip)
  label: string;
  win_rate: number;         // fraction of rollouts that cleared the act
  wins: number;
  n: number;                // rollouts
  avg_final_hp: number;
  avg_floors_advanced: number;
  avg_hp_at_deepest: number;
  value: number;            // composite ranking score
  delta: number;            // value minus the best other option
  is_best: boolean;
  inconclusive: number;
}

/** Result of POST /api/live/card-advisor (EV-based card-pick advisor). */
export interface CardAdvice {
  candidates: AdvisorCandidate[];
  act?: number;
  config?: Record<string, unknown>;
  error?: string;
}

/** One option's fast combat-MCTS ranking for an in-combat card-select. */
export interface ChoiceRankRow {
  key: string; // "choice:<option index>" | "choice:skip"
  label: string;
  win_prob: number | null; // null when the search never visited this option
  q: number | null;
  visits: number;
  is_best: boolean;
}

/** Result of POST /api/live/choice-ranker — one combat MCTS search ranking
 *  every option of the live run's in-combat single-pick card-select (skill
 *  potion picks, Headbutt put-back, upgrade-in-hand, …). This is the SAME
 *  search the deployed run-AI performs at that decision in training/arena
 *  play, so the chips show the agent's actual deliberation. */
export interface ChoiceRanker {
  candidates: ChoiceRankRow[];
  decision?: string;
  act?: number;
  config: Record<string, unknown>;
  error?: string;
}

/** One offered option's PPO run-policy probability (off-combat). */
export interface PpoOption {
  action: string;
  args: Record<string, unknown>;
  label: string;
  prob: number | null;
  is_best: boolean;
}

/** Result of POST /api/live/ppo-policy — the PPO net's masked softmax + V(s)
 *  over the current non-combat decision. */
export interface PpoPolicyPayload {
  ppo_available: boolean;
  impl: string;
  surface: string;
  value: number | null;
  options: PpoOption[];
  error: string | null;
}

/** Resolved two-axis run-mode config (off-combat brain x combat budget) —
 *  the shared shape for POST /api/live/run-mode's request body and response. */
export interface RunModeConfig {
  off_combat_brain: string; // "heuristic" | "ppo"
  ppo_impl: string; // "rust" | "python"
  combat_sims: number;
  combat_boss_sims: number;
  combat_threads: number;
  reward_sims: number | null;
}
export type RunModePayload = RunModeConfig;

/** A progress event streamed by POST /api/live/card-advisor/stream. */
export type AdvisorProgress =
  | {
      type: "start";
      total: number; // total playouts across all options
      rollouts: number; // playouts per option
      act: number;
      candidates: { idx: number | null; label: string }[];
    }
  | {
      type: "tick";
      done: number; // playouts finished so far
      total: number;
      candidate_idx: number | null; // option this playout belonged to
    };

/** One option's Monte-Carlo act outcome from the generic decision advisor.
 *
 * The sibling of {@link AdvisorCandidate} for shops / events / in-combat card
 * selects. Keyed by a stable string `key` (`"choice:0"` / `"card:1"` /
 * `"relic:0"` / `"option:2"` / `"leave"` / `"choice:skip"`) the rendering
 * component joins its options on — a single decision can mix index spaces. */
export interface DecisionAdvisorCandidate {
  key: string;
  label: string;
  win_rate: number;        // fraction of rollouts that cleared the act
  wins: number;
  n: number;               // rollouts
  avg_final_hp: number;
  avg_floors_advanced: number;
  avg_hp_at_deepest: number;
  value: number;           // composite ranking score
  delta: number;           // value minus the best other option
  is_best: boolean;
  inconclusive: number;
}

/** Result of POST /api/live/decision-advisor. */
export interface DecisionAdvice {
  candidates: DecisionAdvisorCandidate[];
  decision?: string;       // "card_select" | "shop" | "event_choice"
  act?: number;
  config?: Record<string, unknown>;
  error?: string;
}

/** A progress event streamed by POST /api/live/decision-advisor/stream. */
export type DecisionAdvisorProgress =
  | {
      type: "start";
      total: number;
      rollouts: number;
      act: number;
      decision: string;
      candidates: { key: string; label: string }[];
    }
  | {
      type: "tick";
      done: number;
      total: number;
      candidate_key: string;
    };

/** POST /api/live/last-choice-grade: the run-level choice that produced the
 *  active node, graded by agreement with the PPO policy. `prob`s are policy
 *  probabilities, NOT win chances — never render them as "win %". */
export interface ChoiceGrade {
  node_id: number;
  parent_id: number;
  decision: string;
  kind: string;            // "Card reward", "Map", "Event", …
  verdict: "agree" | "tossup" | "flag";
  strength: "strong preference" | "preference";
  played: { label: string; prob: number; rank: number };
  top: { label: string; prob: number };
  alternatives: { label: string; prob: number }[];
  n_options: number;
  checkable: boolean;      // a rollout check can compare you vs the AI
}

/** Result of the choice coach's rollout check: rows keyed "you" / "ai". */
export interface ChoiceCheck {
  candidates: DecisionAdvisorCandidate[];
  act?: number;
  config?: Record<string, unknown>;
  error?: string;
}

/** A progress event streamed by POST /api/live/last-choice-check/stream. */
export type ChoiceCheckProgress =
  | { type: "start"; total: number; rollouts: number; act: number; candidates: { key: string; label: string }[] }
  | { type: "tick"; done: number; total: number; candidate_key: string };

/** One combat decision diffed against the MCTS line (regret review). */
export interface DecisionRegret {
  idx: number;
  turn: number | null;
  floor: number | null;
  act: number | null;
  played_label: string;
  played_win_prob: number;
  best_label: string;
  best_win_prob: number;
  regret: number;          // 0..1
  drop_pct: number;        // round(regret * 100)
  player_rank: number;     // 0 = took the top line
  n_actions: number;
  visits: number;
  matched_best: boolean;
  is_blunder: boolean;
  message: string;
  played_mean_hp: number | null;
  best_mean_hp: number | null;
  hp_drop: number | null;  // >= 0; null when HP data is thin
  severity: number;
  verdict: "best" | "tie" | "slip" | "blunder" | "uncertain";
  best_line: string[];     // the AI's labels for the rest of its turn (may be empty)
  node_id?: number | null; // live runs only: the branch-tree node to seek to (stable across rewinds)
  error?: string;          // present on the live last-move payload when scoring is unavailable
}

/** Result of POST /api/regret-review (post-combat learning engine). */
export interface RegretReview {
  decisions: DecisionRegret[];
  blunders: DecisionRegret[];  // the costliest 1-2 plays
  summary: {
    total_decisions: number;
    blunder_count: number;
    cumulative_drop_pct: number;
    worst_drop_pct: number;
    cumulative_hp_drop: number;
  };
  config?: Record<string, unknown>;
}

/** One scored run-level decision on the WPA value curve (issue #201).
 *  `value` is the PPO critic's shaped-return estimate Φ — a relative outlook
 *  score, NOT a win probability; never render it as a percentage. */
export interface WpaPoint {
  node_id: number | null; // live branch-tree node (seek target); null on replay steps
  idx: number;
  decision: string;
  floor: number | null;
  act: number | null;
  value: number;
  delta_prev: number;
  taken_label: string;
}

/** A decision flagged for the outlook swing it preceded (WPA run review). */
export interface WpaTurningPoint {
  node_id: number | null;
  idx: number;
  decision: string;
  floor: number | null;
  act: number | null;
  taken_label: string;
  value_before: number;
  value_after: number;
  swing: number;
  message: string;
}

/** What changed in the master deck between two autobattle stops. */
export interface AutobattleDeckDelta {
  added: string[];
  removed: string[];
  upgraded: string[];
}

/** The "since your last stop" change-diff (see python/workbench/autobattle.py).
 *  Delta fields are null at the first stop of a run, where only absolutes are
 *  meaningful. */
export interface AutobattleDiff {
  from_floor: number | null;
  to_floor: number | null;
  autoplayed: number;
  hp: number | null;
  max_hp: number | null;
  hp_delta: number | null;
  gold: number | null;
  gold_delta: number | null;
  deck_size: number;
  deck: AutobattleDeckDelta;
  relics_gained: string[];
  relics_lost: string[];
  potions_gained: string[];
  potions_lost: string[];
  lines: string[];
}

/** Result of POST /api/live/autobattle/advance — the SKIP path: the whole leg
 *  resolves server-side and only its far end comes back, so the fights are never
 *  seen (`AutobattleStep` is the watch path). `stopped` false means the
 *  autobattle ended (run over) or hit its auto-play cap rather than reaching a
 *  decision. */
export interface AutobattleAdvance {
  step: Step;
  diff: AutobattleDiff;
  autoplayed: number;
  stopped: boolean;
  over: boolean;
  won: boolean;
  error: string | null;
}

/** Result of POST /api/live/autobattle/step — the WATCH path: one paintable
 *  decision per call, so the auto-played fights appear on screen instead of
 *  being inferred from the diff's HP line. `played` is false when the run was
 *  already sitting on a stop, and `diff` is null until a stop is reached (there
 *  is no window to summarize mid-fight). */
export interface AutobattleStep {
  step: Step;
  diff: AutobattleDiff | null;
  played: boolean;
  stopped: boolean;
  over: boolean;
  won: boolean;
  error: string | null;
  search: LastSearch | null;
  debug: FaithfulDebug | null;
}

/** Verdict bands. `near_tie` is a disagreement the policy barely cared about —
 *  reporting a coin flip as a miss would train the player to match noise. */
export type AutobattleBand = "matched" | "near_tie" | "differed";

/** Result of GET /api/live/autobattle/verdict — agreement with the champion ONLY.
 *  There is deliberately no win-probability field: the critic's value is a
 *  shaped return, not a win probability.
 *
 *  `margin` is the champion's policy probability minus yours (null when the
 *  brain exposes no probabilities, which collapses the bands back to two).
 *  A small margin means the policy was nearly INDIFFERENT — never render it as
 *  "the options are equally good", since an uncertain or undertrained surface
 *  looks exactly the same from the probabilities. */
export interface AutobattleVerdict {
  available: boolean;
  reason?: string;
  agreed?: boolean;
  band?: AutobattleBand;
  margin?: number | null;
  champion_prob?: number | null;
  your_prob?: number | null;
  champion_label?: string;
  your_label?: string;
  brain?: string;
}

/** Result of POST /api/wpa-review (+ its live twin): the run-level review. */
export interface WpaReview {
  available: boolean;
  reason?: string;
  points: WpaPoint[];
  turning_points: WpaTurningPoint[];
  summary: {
    n_points?: number;
    first_value?: number;
    final_value?: number;
    biggest_drop?: number;
    biggest_gain?: number;
    n_turning_points?: number;
    won?: boolean;
  };
  caveats: string[];
}

/** One arm of a WPA counterfactual spot check (agent-played continuation). */
export interface WpaVerifyArm {
  label: string;
  won: boolean;
  over: boolean;
  steps: number;
  error: string | null;
}

/** Result of POST /api/live/wpa-verify: played line vs the PPO's alternative,
 *  both agent-finished once from the flagged decision (a spot check, not a
 *  probability). */
export interface WpaVerify {
  node_id: number;
  decision: string;
  played: WpaVerifyArm;
  alternative: WpaVerifyArm;
  verdict: "same" | "alternative_better" | "played_better";
  note: string;
}

/** Snapshot of the current workbench moment, returned by GET /api/bugreport/:id. */
export interface BugReportDump {
  schema_version: number;
  meta: Record<string, unknown>;
  repro: Record<string, unknown>;
  state: Record<string, unknown>;
  ui: { screen: string | null; overlay: string | null };
  note: string | null;
}

/** One point on the run's axis-score trend (a deck change). */
export interface TrendPoint {
  idx: number;
  floor: number | null;
  act: number | null;
  deck_size: number;
  scores: AxisScores;
}

// Base URL the backend serves art from. Defaults to the locally-mounted
// `/assets` route and is overwritten by whatever the backend reports (a GCS
// bucket / CDN origin in lean deployments — see STS2_ASSETS_BASE_URL). Every
// step response carries it, so it is captured centrally in `json` below rather
// than threaded through the component tree. The static (browser-only) build
// has no backend to report one, so it starts from VITE_ASSETS_BASE instead.
let assetsBase = configuredAssetsBase();

/** Build a URL for an art file, given its path relative to the art root. */
export function assetUrl(rel: string): string {
  return `${assetsBase}/${rel}`;
}

/** The current art base (`/assets` or an offloaded origin), for the placeholder listener. */
export function currentAssetsBase(): string {
  return assetsBase;
}

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const body = (await r.json()) as T;
  const base = (body as { assets_base?: unknown } | null)?.assets_base;
  if (typeof base === "string" && base) assetsBase = base.replace(/\/$/, "");
  return body;
}

/** One node in the live branch tree (a recorded decision point). */
/**
 * One line of the game-style combat log, already localized by the server
 * (`workbench/combatlog.py`).
 *
 * `text` always renders the whole line; the split-out `actor` / `target` /
 * `amount` / `source` exist so the panel can colour the names the way the
 * game's own log does, without re-parsing the sentence.
 */
export interface CombatLogRow {
  kind: string;          // "damage" | "block" | "power" | "power_flash" | "monster_move" | ...
  round: number | null;
  side: "player" | "enemy" | null;  // whose TURN it was, not whose action
  text: string;
  // The verb the line reads with ("dealt" | "gained" | "applied" | "used" | …).
  // Decided server-side: several kinds have two shapes the client cannot tell
  // apart from the fields alone (a self-applied power has actor === target and
  // still reads "applied").
  verb: string | null;
  actor: string | null;
  actor_side: "player" | "enemy" | null;
  target: string | null;
  target_side: "player" | "enemy" | null;
  amount: number | null;
  noun: string | null;   // Block / the power / the orb / the move
  source: string | null; // the card or move it came from ("using Strike")
  /** Creature uids for the actor / target, for joining a row to a board slot.
   *  `null` for the player (who has no monster uid) and for rows with no
   *  actor. See `effects/uidTargets.ts`. */
  actor_uid?: number | null;
  target_uid?: number | null;
  /** The STABLE identity of the card or move this row came from — `source` is
   *  the prettified display name and cannot be joined against. Used by the VFX
   *  binding table (`effects/bindings.ts`). */
  source_kind?: "card" | "move" | null;
  source_card?: string | null;
  source_upgraded?: boolean | null;
  source_move_id?: string | null;
  /** Raw monster id (`"Crusher"`), for moves only — a `source_move_id` alone
   *  is not unique across monsters, so this is the other half of the VFX
   *  binding key. `null` for card sources. Left PascalCase, not prettified —
   *  it is a join key, not display text. */
  source_monster?: string | null;
  /** `damage` rows only. `amount` is the UNBLOCKED damage; `blocked` is what
   *  block absorbed, so `amount === 0 && blocked > 0` is a fully-blocked hit.
   *  `hp_after === 0` is the only death signal the engine log carries — its
   *  `EventLogEntry` enum has no death entry. */
  blocked?: number | null;
  hp_after?: number | null;
}

export interface LiveNode {
  id: number;
  parent_id: number | null;
  idx: number;          // depth from root
  decision: string;
  label: string;
  details: string[];
  // THIS step's slice of the combat log (not the whole fight — the panel
  // concatenates the slices along the active path, so seeking back truncates
  // the log to that line). Absent on backends without the engine log.
  combat_log?: CombatLogRow[];
  error: boolean;
  child_ids: number[];  // >1 means this is a fork point
  floor?: number | null; // map/run floor this step sits on (carried forward)
  act?: number | null;   // 1-based act index this step sits on (carried forward)
  // Combat round of this node's (post-action) state; null off-combat. The
  // round a node's action was TAKEN in is its parent's turn — the history
  // ledger groups by that.
  turn?: number | null;
  // Live-GAME trees only (/api/bridge/tree): the mod's snapshot backing this
  // node. Undefined on workbench-local sessions (those are always seekable).
  snapshot_id?: string | null;
  seekable?: boolean;
}
export interface LiveTimeline {
  nodes: LiveNode[];
  active_path: number[];   // root → active leaf, by id
  active_leaf: number | null;
}

/** One node in the agent's MCTS search tree (from /api/search_tree). */
export interface SearchTreeNode {
  id: number;
  parent: number | null;
  visits: number;
  q: number;            // value_sum / visits (player-perspective EV in [-1,1])
  value_sum: number;
  wins: number;
  losses: number;
  ties: number;
  is_terminal: boolean;
  proven_win: boolean;
  proven_loss: boolean;
  children: number[];
  // One entry per child, in `children` order: the action on that edge and the
  // traffic through it. The search graph is a DAG (transpositions are merged),
  // so a node can hang under several parents via different actions — the
  // label and visits of an edge belong to the edge, not to the child. Absent
  // in older dumps; readers fall back to the child's own fields.
  edges?: SearchTreeEdge[];
  // The action from this node's allocating `parent` (null at the root). For a
  // shared node this is only one of its incoming edges — prefer the parent's
  // `edges` when walking from the root. A chance OUTCOME's kind is "outcome".
  kind: string | null;          // "play" | "end_turn" | "potion" | "channel" | "outcome" | ...
  label: string | null;
  target_index: number | null;
  // ── Chance-node search ───────────────────────────────────────────────────
  // The node's own type. "chance" nodes are nature resolving a stochastic
  // transition: their `children` are sampled outcomes, `q` is the expectimax
  // E[V] (not a max), and each child's probability = child.visits/this.visits.
  node_kind?: "decision" | "chance";
  // For a chance node: the stochastic action it resolves (e.g. "End turn").
  chance_action?: string | null;
  // For a chance node: the DPW outcome cap at current visits,
  // ceil(C·(visits+1)^alpha). Compare to children.length to see if the
  // outcome distribution is still being widened / is truncated.
  outcome_cap?: number;
  chance_draws?: number;        // total widening draws taken (reseed ordinal)
  chance_closed?: boolean;      // outcome set declared effectively enumerated
  // For a node reached as a chance outcome: its outcome ordinal.
  outcome_ordinal?: number | null;
  // State summary: the outcome after this node's action.
  summary?: {
    player_hp: number;
    player_max_hp: number;
    player_block: number;
    energy: number;
    round: number;
    enemies: { uid: number; hp: number; max_hp: number }[];
  };
}
/** One parent→child edge of the search graph (see SearchTreeNode.edges). */
export interface SearchTreeEdge {
  child: number;
  kind: string | null;
  label: string | null;
  target_index: number | null;
  outcome_ordinal?: number | null;
  // Decision edge: simulations that took this edge FROM this parent, n(s,a).
  // Outcome edge: the outcome child's visits (what the engine samples by).
  visits: number;
}
export interface SearchTree {
  nodes: SearchTreeNode[];
  root: number | null;
  pv: Record<string, unknown>[];
  sims?: number;
  c_puct?: number;
  error?: string;
}

/** Metadata header for a named save (the load-dialog row shape; no action line). */
export interface SaveMeta {
  name: string;
  saved_at: string; // UTC ISO-8601
  character: string;
  seed: string;
  ascension: number;
  progress: { act: number | null; floor: number | null; hp: number | null; max_hp: number | null };
  step_count: number;
}

/** Which half of a pasted live state to rebuild. "auto" prefers the fight. */
export type ImportLane = "auto" | "combat" | "run";

/** What an import reproduces exactly, and what it does not. */
export interface ImportFidelity {
  exact: string[];
  /** True when the run's RNG cursors were restored — without them, every
   *  downstream reward/shop/event roll is this engine's, not the live run's. */
  rng_restored: boolean;
  warnings: string[];
  surface: string;
  source: string;
  lane?: ImportLane;
}

/** A parsed paste, described without touching the current run. */
export interface ImportPreview {
  surface: string;
  source: string;
  has_combat: boolean;
  lanes: ImportLane[];
  run: {
    seed: string;
    character: string;
    ascension: number;
    act: number;
    floor: number;
    hp: number;
    max_hp: number;
    gold: number;
    deck_size: number;
    relics: string[];
    potions: string[];
    coord: [number, number] | null;
  };
  fidelity: ImportFidelity;
}

/** Tunable MCTS knobs forwarded to the search (maps to RecommendConfig). */
export interface MctsConfig {
  sims: number;
  c_puct: number;       // UCT exploration constant (higher = more exploration)
  depth: number;        // rollout simulation depth (plies)
  rollout_policy: "heuristic" | "uniform";
  seed: number;         // RNG seed — change to re-roll determinizations
  threads: number;      // root-parallel workers; budget is SPLIT across them (faster, same total sims)
  /** Hidden-information mode. "beliefdraw" is what the deployed agent runs:
   *  draws are chance nodes, so the search plans against a draw order it does
   *  not know. "clairvoyant" lets the root see the true draw order — fine for
   *  engine work, wrong for advising a human, who cannot see it either. */
  hidden_info: "beliefdraw" | "clairvoyant" | "drawchance" | "rootscramble";
  /** Draw-node progressive-widening constant. Only meaningful under
   *  "beliefdraw"; 2.0 is the F2-retuned deployed value and travels with
   *  c_puct 2.5 as a set. */
  draw_widening_c: number;
  /** Drop follow-up card plays whose reverse order the engine proves at least
   *  as good (`MctsConfig::dominance_pruning`): "Strike, then Beam Cell" is
   *  pruned because "Beam Cell, then Strike" deals more damage and is
   *  otherwise identical. Optional so older callers still type-check; absent
   *  means off. */
  dominance_pruning?: boolean;
}

export const DEFAULT_MCTS_CONFIG: MctsConfig = {
  // One budget for BOTH surfaces. The search-tree inspector shipped 1000 while
  // the hint panel overrode this to 2000, so the two ran different-strength
  // searches over the same position by default and disagreed for a reason no
  // visible knob explained. Either panel's `sims` selector still overrides it.
  sims: 2000,
  // 2.5 is the value the book sweep settled on (logs/2026-08-12-cpuct-sweep)
  // and what the deployed agent runs. The panel shipped 1.4 — a combination
  // nobody swept — purely because that was the recommender's own default.
  c_puct: 2.5,
  // Rollout ply cap. Must be deep enough for a rollout to reach a real
  // terminal (win/lost); a too-shallow cap truncates every Act-1 rollout
  // so every action scores ~0 win-prob (flat) and the recommendation
  // collapses to "end turn". 200 matches the engine's own default.
  depth: 200,
  rollout_policy: "heuristic",
  seed: 0,
  // Root-parallel workers. ONE tree by default: >1 splits the sim budget across
  // independent shallower trees and merges their root counters, which evens out
  // the visit distribution the sidebar draws as "search %" (measured: 16 workers
  // at 2000 sims gave root visits [401,401,401,401,396] where 1 worker gave
  // [430,430,421,408,311]) and leaves no single tree to inspect. It also made
  // the hint panel structurally incomparable to the search-tree inspector, which
  // is single-tree by construction (`mcts_search_tree` has no threads argument).
  // Raise it when you want wall-clock over a legible search.
  threads: 1,
  // Match the deployed honest search. These three travel together — c_puct
  // 2.5 + beliefdraw + draw_widening_c 2.0 — and the engine adds the run
  // agent's EndTurn widening (5.787 / 0.436) for beliefdraw by itself
  // (`MctsConfig::apply_mode_dialect`, GH #758), so it is not repeated here.
  // `sts2-workbench/tests/search_dialect_parity.rs` parses this object and
  // fails if it stops searching like the run agent.
  // It matches the run agent's BASE dialect on purpose (GH #830): the run
  // agent's Ironclad layer (knobs + BC move prior) is not taken here.
  hidden_info: "beliefdraw",
  draw_widening_c: 2.0,
  // Workbench-only on purpose: the run agent does not prune. It cuts the
  // dominated move being the top suggestion from 14.7% to 10.6% of searches
  // on 2800 engine-proven ordering pairs, at an unchanged fight win rate
  // (77.40% vs 77.38%, 9000 paired A10 fights). Not a dialect field, so
  // `search_dialect_parity.rs` still holds.
  dominance_pruning: true,
};

/** Below this, the #1 row's Q lead over #2 is inside the search's own
 *  cross-seed dispersion, so the ranking is a coin flip and the panel says so
 *  instead of printing a confident ★.
 *
 *  Calibrated, not guessed: 70 positions across all five characters, 5 seeds
 *  each at the deployed config, gave a median cross-seed SD of the top-2 gap
 *  of 0.0021. This is 2x that. It fires on ~33% of positions, which sits
 *  inside the "three quarters of decisions are statistically unresolved"
 *  finding in GH #442 measured at a looser bar.
 *
 *  It does NOT catch every wrong recommendation. A ranking can be perfectly
 *  reproducible and still wrong (see bug report wb-e34ef8b6039bc0f3, whose
 *  gap of ~0.010 is well clear of this bar and held on 20/20 seeds). This
 *  flags "the search doesn't know", not "the search is mistaken". */
export const UNRESOLVED_TOP2_Q_GAP = 0.004;

/** One recorded MCTS run, summarized for the runs table. */
export interface MctsRunSummary {
  stem: string;
  won: boolean;
  final_turn: number | null;
  /** Floor reached — full-run logs only; null for a recorded single combat. */
  final_floor: number | null;
  turn1_q: number | null;
  enemy_min_pct: number | null;
  path: string;
}
/** Aggregate loss-pattern dashboard across the recorded MCTS runs. */
export interface MctsAggregate {
  n_runs: number;
  n_losses: number;
  final_turn_histogram: Record<string, number>;
  turn1_q: { min: number | null; max: number | null; mean: number | null };
  pattern_tags: Record<string, number>;
  representative_losses: {
    stem: string;
    final_turn: number | null;
    turn1_q: number | null;
    enemy_min_pct: number | null;
  }[];
}
export const fetchMctsRuns = () =>
  apiFetch("/api/mcts-runs").then(json<{ runs: MctsRunSummary[] }>);
export const fetchMctsAggregate = () =>
  apiFetch("/api/mcts-runs/aggregate").then(json<MctsAggregate>);

/** Feature flags a deployment reports alongside its engines. Absent (older
 *  backends) means everything is available; the browser-only WASM build
 *  reports `advisors: false, run_mode: false` and the SPA hides those
 *  controls instead of offering calls that can only answer 501. */
export interface CapabilityFeatures {
  advisors?: boolean;
  run_mode?: boolean;
  saves?: boolean;
  replay?: boolean;
  /** The engine has no upload target for bug reports; the browser build
   *  turns the button on itself when its host runs the Pages Function
   *  (src/bugreport.ts `bugReportEndpointAvailable`). */
  bug_reports?: boolean;
  /** The state-import graft is not in the browser build yet. */
  state_import?: boolean;
  /** The SimLab position sandbox (editor + per-move MCTS). The browser build
   *  answers `/api/simlab/*` from the wasm since M5; `simlab_sweeps` covers
   *  the distribution study half, which is still desktop-only. */
  simlab?: boolean;
  simlab_sweeps?: boolean;
}

export const api = {
  // Which game engines this deployment can actually run. The deployed image is
  // C#-free (no .NET runtime), so the engine dropdown is built from this rather
  // than hardcoded -- offering an engine the server cannot spawn turned "Start
  // run" into a no-op.
  capabilities: () =>
    apiFetch("/api/capabilities").then(json<{
      backends: string[];
      features?: CapabilityFeatures;
      /** GH #801: the game builds this engine can run; absent = v0.107.1 only. */
      game_versions?: string[];
      default_game_version?: string;
    }>),
  runs: () => apiFetch("/api/runs").then(json<{ runs: string[] }>),
  runStat: (path: string) =>
    apiFetch(`/api/runs/stat?path=${encodeURIComponent(path)}`).then(
      json<{ size: number; mtime: number }>,
    ),
  // Live-GAME bridge (the SpireMentor mod, via the advisor's control plane):
  // status probe, branch tree, rewind, companion search tree. Through the
  // server's same-origin proxy — or, in the browser build with a control
  // plane URL configured (src/bridge.ts, GH #709), straight to it: the
  // proxy's routes map 1:1 onto the control plane's, and its one transform
  // (rendering each node's raw combat log with the loc map) is the wasm's
  // `POST /api/bridge/render-tree`.
  bridge: {
    status: () =>
      (bridgeBase() ? bridgeFetch("/status") : apiFetch("/api/bridge/status")).then(
        json<{
          connected: boolean;
          character?: string | null;
          seed?: string | null;
          floor?: number | null;
          act?: number | null;
          log_path?: string | null;
        }>,
      ),
    tree: () =>
      bridgeBase()
        ? bridgeFetch("/tree")
            .then(json<LiveTimeline>)
            .then((tree) =>
              apiFetch("/api/bridge/render-tree", {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({ tree }),
              }).then(json<LiveTimeline>),
            )
        : apiFetch("/api/bridge/tree").then(json<LiveTimeline>),
    snapshots: () =>
      (bridgeBase() ? bridgeFetch("/snapshots") : apiFetch("/api/bridge/snapshots")).then(
        json<{
          run_start_time: number;
          snapshots: {
            id: string;
            seq: number;
            floor: number;
            act: number;
            post_combat: boolean;
          }[];
        }>,
      ),
    /** The live game's latest decision position: the mod's Copy state
     *  message, which the state import takes as is. 404 before the first. */
    state: () =>
      (bridgeBase() ? bridgeFetch("/state") : apiFetch("/api/bridge/state")).then(
        json<Record<string, unknown>>,
      ),
    seek: (nodeId?: number, snapshotId?: string) => {
      const init: RequestInit = {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ node_id: nodeId ?? null, snapshot_id: snapshotId ?? null }),
      };
      return (bridgeBase() ? bridgeFetch("/restore", init) : apiFetch("/api/bridge/seek", init)).then(
        json<{ snapshot_id?: string; floor?: number }>,
      );
    },
    searchTree: (sims?: number) => {
      const init: RequestInit = {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ sims }),
      };
      return (bridgeBase() ? bridgeFetch("/search_tree", init) : apiFetch("/api/bridge/search-tree", init)).then(
        json<SearchTree>,
      );
    },
  },
  load: (path: string) =>
    apiFetch("/api/load", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ path }),
    }).then(json<{ count: number; resume?: ResumeCapability }>),
  step: (idx: number) => apiFetch(`/api/step/${idx}`).then(json<Step>),
  deckAnalysis: (idx: number) =>
    apiFetch(`/api/step/${idx}/deck-analysis`).then(json<DeckAnalysis>),
  deckTrend: () => apiFetch("/api/deck/trend").then(json<{ points: TrendPoint[] }>),
  // `gameVersion` (GH #877): the build whose card tables grade the cards;
  // sent only when it is not the default, like every other route.
  deckRewardPreview: (deck: unknown[], candidates: unknown[], gameVersion?: string | null) =>
    apiFetch("/api/deck/reward-preview", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ deck, candidates, ...gameVersionBody(gameVersion) }),
    }).then(json<RewardPreview>),
  timeline: () => apiFetch("/api/timeline").then(json<{ steps: TimelineEntry[] }>),
  map: () => apiFetch("/api/map").then(json<{ rows: MapRow[]; graphs: MapGraph[] }>),
  coach: () => apiFetch("/api/coach").then(json<CoachLog>),
  recommend: (idx: number, config: Record<string, unknown>) =>
    apiFetch("/api/recommend", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ idx, config }),
    }).then(json<{ scores: Score[]; pv: string[] | null }>),
  searchTree: (idx: number, config: Record<string, unknown>) =>
    apiFetch("/api/search_tree", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ idx, config }),
    }).then(json<SearchTree>),
  // Post-combat regret review for the whole loaded run: diff each move against
  // the MCTS line and surface the costliest plays. `recompute` re-runs MCTS.
  regretReview: (config: Record<string, unknown>) =>
    apiFetch("/api/regret-review", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ config }),
    }).then(json<RegretReview>),
  // WPA run review of the loaded replay: the run-level value curve + turning
  // points recorded while the run was played live (issue #201).
  wpaReview: (config: Record<string, unknown>) =>
    apiFetch("/api/wpa-review", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ config }),
    }).then(json<WpaReview>),
  live: {
    start: (body: {
      character: string;
      seed?: string;
      ascension?: number;
      backend?: string;
      /** GH #544: "overgrowth" | "underdocks" | "roll" (the real lobby pick). */
      act1?: string;
      /** GH #801: "v0.107.1" (default) | "v0.111.0". */
      game_version?: string;
    }) =>
      apiFetch("/api/live/start", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      }).then(json<Step>),
    // Start a run that reproduces the offline benchmark
    // (`play_full_run_rust.play_run_mcts`, the 2000-seed win-rate sweep's
    // driver) move-for-move: rust backend, integer seed, no interactive UI
    // flags. Follow with `runModeStep` to run it out with the benchmark policy.
    startFaithful: (body: {
      character: string;
      seed: string;
      ascension?: number;  // run ascension level (0–20); default 0
      sims?: number;       // MCTS iters in regular monster combat
      boss_sims?: number;  // MCTS iters in boss combat
      reward_sims?: number; // boss-sim depth per card-reward candidate
      threads?: number;
      /** GH #801: "v0.107.1" (default) | "v0.111.0". */
      game_version?: string;
    }) =>
      apiFetch("/api/live/start-faithful", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      }).then(json<Step>),
    // Take over the loaded recorded run's combat at step `idx` ("play from
    // here") — interactive combat from the exact state, with branch tree + hints.
    fromReplay: (idx: number) =>
      apiFetch("/api/live/from_replay", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ idx }),
      }).then(json<Step>),
    state: () => apiFetch("/api/live/state").then(json<Step>),
    // Current act's full map graph, on demand — available at ANY decision
    // (combat, card reward, shop, …), not just `map_select`. `graphs` is empty
    // when no map exists (e.g. the boss room).
    map: () => apiFetch("/api/live/map").then(json<{ rows: MapRow[]; graphs: MapGraph[] }>),
    deckAnalysis: () => apiFetch("/api/live/deck-analysis").then(json<DeckAnalysis>),
    action: (action: string, args: Record<string, unknown>) =>
      apiFetch("/api/live/action", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action, args }),
      }).then(json<{ step: Step; error: string | null; label: string; details: string[] }>),
    hint: (config: Record<string, unknown>) =>
      apiFetch("/api/live/hint", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      }).then(json<{ scores: Score[]; pv: string[] | null }>),
    // The policy's recommended (verb, args) for the current decision: a
    // run-level one (card reward, shop, event, map, rest, …) or an open
    // in-combat card choice, which the combat MCTS answers at the same config
    // as that screen's chips. Resolves to null in plain combat / at game over /
    // on a multi-pick in-combat choice / on a backend that can't recommend
    // (409), so the caller falls back to the MCTS line or its own default.
    autoAction: async (): Promise<{ action: string; args: Record<string, unknown> } | null> => {
      const resp = await apiFetch("/api/live/auto-action", { method: "POST" });
      if (resp.status === 409) return null;
      return json<{ action: string; args: Record<string, unknown> }>(resp);
    },
    // Update the two-axis run mode (off-combat brain x combat budget).
    runMode: async (cfg: Partial<RunModeConfig>): Promise<RunModePayload> => {
      const resp = await apiFetch("/api/live/run-mode", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(cfg),
      });
      return json<RunModePayload>(resp);
    },
    // One unified stepper decision (loop this to play a run out visibly).
    //
    // `search` is the combat MCTS root distribution for the move this step
    // applied — null off combat. Its rows belong to the position BEFORE the
    // returned `step`, which is why the server keeps them out of `step.scores`
    // (those are index-matched onto the rendered hand). See searchPanel.ts.
    runModeStep: () =>
      apiFetch("/api/live/run-mode-step", { method: "POST" }).then((r) =>
        json<{
          step: Step;
          done: boolean;
          over: boolean;
          victory: boolean;
          error: string | null;
          debug: FaithfulDebug | null;
          search: LastSearch | null;
        }>(r),
      ),
    // Autobattle mode: auto-play combat until the next RUN-level decision, the
    // whole leg in one call (the fights are not seen; `autobattleStep` watches).
    // The champion's pick is deliberately absent from this payload — fetch it
    // from `autobattleVerdict` after committing, never before (see AutobattlePanel).
    autobattleAdvance: () =>
      apiFetch("/api/live/autobattle/advance", { method: "POST" }).then(json<AutobattleAdvance>),
    // The same leg, one visible decision at a time — the caller loops this and
    // paints each step, which is how the fights are watched rather than skipped.
    autobattleStep: () =>
      apiFetch("/api/live/autobattle/step", { method: "POST" }).then(json<AutobattleStep>),
    // Reveal the champion's pick for the decision just committed.
    autobattleVerdict: () => apiFetch("/api/live/autobattle/verdict").then(json<AutobattleVerdict>),
    // PPO probability panel; null when the decision isn't PPO's (combat, …).
    ppoPolicy: async (): Promise<PpoPolicyPayload | null> => {
      const resp = await apiFetch("/api/live/ppo-policy", { method: "POST" });
      if (resp.status === 409) return null;
      return json<PpoPolicyPayload>(resp);
    },
    searchTree: (config: Record<string, unknown>) =>
      apiFetch("/api/live/search_tree", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      }).then(json<SearchTree>),
    // Fast per-option combat-MCTS ranking for an in-combat single-pick
    // card-select — one search (the same one the deployed agent runs there),
    // cheap enough to fetch automatically when the choice opens. 409 outside
    // an in-combat card-select.
    choiceRanker: (config: Record<string, unknown>) =>
      apiFetch("/api/live/choice-ranker", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      }).then(json<ChoiceRanker>),
    // Deep, slow EV analysis: Monte-Carlo each card-reward option (take A/B/C
    // vs skip) over the rest of the act. Run on demand only.
    cardAdvisor: (config: Record<string, unknown>) =>
      apiFetch("/api/live/card-advisor", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      }).then(json<CardAdvice>),
    // Streaming twin of cardAdvisor: reads newline-delimited JSON, calling
    // `onProgress` for each start/tick event and resolving with the final
    // advice. Lets the UI show a real progress bar over the slow rollouts.
    cardAdvisorStream: async (
      config: Record<string, unknown>,
      onProgress: (ev: AdvisorProgress) => void,
    ): Promise<CardAdvice> => {
      const resp = await apiFetch("/api/live/card-advisor/stream", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`${resp.status} ${await resp.text()}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let result: CardAdvice | null = null;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          const ev = JSON.parse(line) as
            | AdvisorProgress
            | (CardAdvice & { type: "result" })
            | { type: "error"; error: string };
          if (ev.type === "result") {
            result = ev; // the extra `type` field is harmless on CardAdvice
          } else if (ev.type === "error") {
            result = { candidates: [], error: ev.error };
          } else {
            onProgress(ev);
          }
        }
      }
      if (result == null) throw new Error("card advisor stream ended without a result");
      return result;
    },
    // Generic sibling of cardAdvisor for shops / events / in-combat card
    // selects: Monte-Carlo each option over the rest of the act. On demand only.
    decisionAdvisor: (config: Record<string, unknown>) =>
      apiFetch("/api/live/decision-advisor", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      }).then(json<DecisionAdvice>),
    // Streaming twin of decisionAdvisor (newline-delimited JSON), mirroring
    // cardAdvisorStream so the slow rollouts drive a live progress bar.
    decisionAdvisorStream: async (
      config: Record<string, unknown>,
      onProgress: (ev: DecisionAdvisorProgress) => void,
    ): Promise<DecisionAdvice> => {
      const resp = await apiFetch("/api/live/decision-advisor/stream", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`${resp.status} ${await resp.text()}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let result: DecisionAdvice | null = null;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          const ev = JSON.parse(line) as
            | DecisionAdvisorProgress
            | (DecisionAdvice & { type: "result" })
            | { type: "error"; error: string };
          if (ev.type === "result") {
            result = ev;
          } else if (ev.type === "error") {
            result = { candidates: [], error: ev.error };
          } else {
            onProgress(ev);
          }
        }
      }
      if (result == null) throw new Error("decision advisor stream ended without a result");
      return result;
    },
    // "Blunder right after you did one": regret of the move that produced the
    // current state, vs the MCTS line for the previous combat position.
    // Choice coach: grade the run-level choice just made (map, card reward,
    // shop, rest, event, …) by policy agreement. null = nothing to grade (a
    // combat move, a forced pick, no net loaded).
    lastChoiceGrade: async (): Promise<ChoiceGrade | null> => {
      const resp = await apiFetch("/api/live/last-choice-grade", { method: "POST" });
      if (resp.status === 409) return null;
      return json<ChoiceGrade>(resp);
    },
    // On-demand rollout check of a graded choice: your pick vs the AI's top
    // pick, paired playouts to the end of the act. Slow — streams progress.
    lastChoiceCheckStream: async (
      nodeId: number,
      config: Record<string, unknown>,
      onProgress: (ev: ChoiceCheckProgress) => void,
    ): Promise<ChoiceCheck> => {
      const resp = await apiFetch("/api/live/last-choice-check/stream", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config: { ...config, node_id: nodeId } }),
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`${resp.status} ${await resp.text()}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let result: ChoiceCheck | null = null;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          const ev = JSON.parse(line) as
            | ChoiceCheckProgress
            | (ChoiceCheck & { type: "result" })
            | { type: "error"; error: string };
          if (ev.type === "result") {
            result = ev;
          } else if (ev.type === "error") {
            result = { candidates: [], error: ev.error };
          } else {
            onProgress(ev);
          }
        }
      }
      if (result == null) throw new Error("choice check stream ended without a result");
      return result;
    },
    // Per-turn coach: grade the combat turn the End turn just closed, as the
    // live advisor's `turn` record (409 = nothing to grade or a silent turn).
    lastTurnGrade: (config: Record<string, unknown>) =>
      apiFetch("/api/live/last-turn-grade", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      }).then(json<CoachTurnRecord & { searches?: number }>),
    lastMoveRegret: (config: Record<string, unknown>) =>
      apiFetch("/api/live/last-move-regret", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      }).then(json<DecisionRegret>),
    // Full-run regret review over the live active line: diff every combat move
    // made so far against the MCTS line and surface the costliest plays. The
    // live twin of the replay `regretReview`.
    regretReview: (config: Record<string, unknown>) =>
      apiFetch("/api/live/regret-review", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      }).then(json<RegretReview>),
    // WPA run review over the live active line — the run-level twin of
    // `regretReview` (map/card/shop/rest decisions instead of combat moves).
    wpaReview: (config: Record<string, unknown>) =>
      apiFetch("/api/live/wpa-review", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ config }),
      }).then(json<WpaReview>),
    // Counterfactual spot check of one WPA turning point. SLOW: the agent
    // finishes the run twice (played line vs the PPO's alternative).
    wpaVerify: (nodeId: number) =>
      apiFetch("/api/live/wpa-verify", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ node_id: nodeId }),
      }).then(json<WpaVerify>),
    timeline: () => apiFetch("/api/live/timeline").then(json<LiveTimeline>),
    seek: (nodeId: number) =>
      apiFetch("/api/live/seek", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ node_id: nodeId }),
      }).then(json<{ step: Step }>),
    quit: () =>
      // `log_path` is null where nothing is written to disk (the browser build).
      apiFetch("/api/live/quit", { method: "POST" }).then(json<{ ok: boolean; log_path: string | null }>),
    save: (name: string) =>
      apiFetch("/api/live/save", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name }),
      }).then(json<{ name: string; path: string; step_count: number }>),
    // Read a state copied out of the live game WITHOUT touching the current
    // run — what the import dialog shows before the user commits.
    importPreview: (payload: string) =>
      apiFetch("/api/live/import-state/preview", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ payload }),
      }).then(json<ImportPreview>),
    // Rebuild it, replacing the live run. `lane` "combat" resumes the exact
    // fight; "run" rebuilds the run position; "auto" prefers the fight.
    // `gameVersion` (GH #801) is sent only when it is not the default, so a
    // 0.107 import posts the body it always did.
    importState: (payload: string, lane: ImportLane = "auto", gameVersion?: string) =>
      apiFetch("/api/live/import-state", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ payload, lane, ...gameVersionBody(gameVersion) }),
      }).then(json<Step & { import: ImportFidelity }>),
    // Launch a SimLab sweep from live board position(s). Naming several
    // branch nodes compares them as distributions, not single playouts.
    simlabSweep: (body: {
      name?: string;
      strategies?: string[];
      n_seeds?: number;
      node_ids?: number[];
    }) =>
      apiFetch("/api/live/simlab/sweep", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      }).then(json<{ sweep_id: string; url: string; cells: string[]; skipped: string[] }>),
    // The active board as a SimLab sandbox position (`simlab-position/1`).
    // What "SimLab from here" carries across in the URL hash.
    position: () =>
      apiFetch("/api/live/position").then(json<{ position: Record<string, unknown>; label: string; notes: string[] }>),
    // SimLab sweep-form fields for the imported run (fights from turn 1).
    simlabScenario: () =>
      apiFetch("/api/live/simlab/scenario").then(json<Record<string, unknown>>),
  },
  simlab: {
    // A pasted state → SimLab sandbox position, without touching the live run.
    // Through `apiFetch`: in the browser build the wasm answers it (GH #708).
    importPosition: (text: string) =>
      apiFetch("/api/simlab/position/import", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text }),
      }).then(json<{ position: Record<string, unknown>; notes: string[]; source: string }>),
  },
  saves: {
    list: () => apiFetch("/api/saves").then(json<{ saves: SaveMeta[] }>),
    load: (name: string) =>
      apiFetch("/api/saves/load", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name }),
      }).then(json<Step>),
  },
  captureBugReport: (body: {
    mode: string; step?: number | null; screen?: string | null;
    overlay?: string | null; note?: string | null; run_path?: string | null;
  }) =>
    isStaticMode()
      ? captureBrowserBugReport(body)
      : apiFetch("/api/bugreport", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }).then(json<{ id: string; url: string }>),
  fetchBugReport: (id: string) =>
    isStaticMode()
      ? fetchBrowserBugReport(id)
      : apiFetch(`/api/bugreport/${id}`).then(json<BugReportDump>),
};
