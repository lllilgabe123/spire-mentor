/**
 * The workbench's command registry — every one-shot action the top bar used
 * to render as its own button, as data.
 *
 * The bar grew to ~25 conditionally rendered buttons in one sideways-scrolling
 * row: navigation, run lifecycle, AI switches, one-shot analyses and app
 * settings all at the same weight, appearing and disappearing as the run
 * moved. This module is the fix's foundation: the Run and Analyze menus, the
 * Ctrl+K command palette and the pinned "Play from here" button are all views
 * of the list `buildCommands` returns, so a new feature is one entry here,
 * not a new button plus a new key handler.
 *
 * Two visibility rules, deliberately different:
 *  - HIDDEN when the command depends on the MODE (play / replay) or on a
 *    feature flag / build. Switching mode is a whole-context change, so a
 *    different menu is expected.
 *  - DISABLED with a reason when it depends on RUN STATE (in combat or not,
 *    run started or not). The menu then keeps its shape while you play — the
 *    old bar's buttons flickered in and out per decision and could not be
 *    learned by position.
 */

export type CommandGroup = "run" | "analyze";

export interface Command {
  id: string;
  /** Rendered label, icon included. */
  label: string;
  group: CommandGroup;
  /** Tooltip when enabled. */
  title?: string;
  /** Present ⇒ shown but not runnable; the text explains why. */
  disabledReason?: string;
  /** Destructive; rendered last in its group, in the HP colour, after a rule. */
  danger?: boolean;
  /** The action is in flight; label gets an ellipsis and the item disables. */
  busy?: boolean;
  run: () => void;
}

export interface CommandActions {
  save: () => void;
  load: () => void;
  importState: () => void;
  importLive: () => void;
  endRun: () => void;
  analyze: () => void;
  regretReview: () => void;
  runReview: () => void;
  searchTree: () => void;
  simlabFromHere: () => void;
  recommend: () => void;
  playFromHere: () => void;
  bridgeTree: () => void;
  liveSearch: () => void;
  bridgeSeek: () => void;
}

export interface CommandContext {
  mode: "play" | "replay" | "mcts";
  hasLive: boolean;
  liveDecision: string | null;
  /** The viewed replay step's decision kind. */
  stepDecision: string | null;
  replayCount: number;
  /** The `advisors` feature flag: every AI opinion surface hangs off it. */
  advisors: boolean;
  stateImport: boolean;
  staticMode: boolean;
  /** The deck analyzer lives in the bar only on short screens (else it is a
   *  board-side button). */
  analyzeInTopbar: boolean;
  resume: { kind: string | null; reason: string | null } | null;
  bridgeConnected: boolean;
  /** The browser build talks to the control plane directly once a URL is
   *  configured (src/bridge.ts); then the live-game views are not tied to
   *  the replay view. */
  bridgeDirect: boolean;
  viewedFloor: number | null;
  busy: Partial<Record<"simlabFromHere" | "recommend" | "playFromHere" | "bridgeSeek", boolean>>;
  actions: CommandActions;
}

const NO_RUN = "no run in progress";
const COMBAT_ONLY = "only during a combat";

export function buildCommands(c: CommandContext): Command[] {
  const out: Command[] = [];
  const a = c.actions;
  const play = c.mode === "play";
  const replay = c.mode === "replay";
  const inCombat = play ? c.liveDecision === "combat_play" : c.stepDecision === "combat_play";

  // ---- run lifecycle ------------------------------------------------------
  if (play) {
    out.push({ id: "save", group: "run", label: "Save", title: "save this run under a name",
      disabledReason: c.hasLive ? undefined : NO_RUN, run: a.save });
    out.push({ id: "load", group: "run", label: "Load", title: "load a saved run", run: a.load });
    if (c.stateImport) {
      out.push({ id: "import-state", group: "run", label: "📋 Import state",
        title: "paste a state copied out of the live game (SpireMentor: Copy state / Ctrl+Shift+C) and play it from here",
        run: a.importState });
      if (c.bridgeConnected) {
        out.push({ id: "import-live", group: "run", label: "🎮 Import from live game",
          title: "load the position the connected game is at right now and play it from here (no copy-paste)",
          run: a.importLive });
      }
    }
    out.push({ id: "end-run", group: "run", label: "End run", title: "end the current run", danger: true,
      disabledReason: c.hasLive ? undefined : NO_RUN, run: a.endRun });
  }
  if (replay && c.replayCount > 0 && c.resume) {
    const reason = c.resume.kind === null
      ? (c.resume.reason ?? "this recording cannot be resumed")
      : c.resume.kind === "combat" && !inCombat
        ? "this recording is a combat fixture — only its combat steps can be taken over"
        : undefined;
    out.push({ id: "play-from-here", group: "run", label: "🎮 Play from here",
      title: "take over the run from this exact point, interactively (branch tree + hints)",
      disabledReason: reason, busy: c.busy.playFromHere, run: a.playFromHere });
  }

  // ---- analyses -------------------------------------------------------------
  if (play && c.analyzeInTopbar) {
    out.push({ id: "analyze", group: "analyze", label: "📊 analyze deck",
      title: "grade this deck: damage / block / velocity",
      disabledReason: c.hasLive ? undefined : NO_RUN, run: a.analyze });
  }
  if (replay) {
    out.push({ id: "recommend", group: "analyze", label: "recommend (H)",
      title: "recompute a fresh MCTS recommendation for the viewed step",
      disabledReason: c.replayCount === 0 ? "no run loaded" : inCombat ? undefined : COMBAT_ONLY,
      busy: c.busy.recommend, run: a.recommend });
  }
  if (c.advisors && (play || replay)) {
    const started = play ? c.hasLive : c.replayCount > 0;
    const notStarted = play ? NO_RUN : "no run loaded";
    out.push({ id: "regret-review", group: "analyze", label: "📉 regret review",
      title: "post-combat regret review: diff every move against the MCTS line and surface the costliest plays",
      disabledReason: started ? undefined : notStarted, run: a.regretReview });
    out.push({ id: "run-review", group: "analyze", label: "📈 run review",
      title: "run review: the AI's outlook curve over the map/card/shop/rest decisions, with the biggest turning points flagged",
      disabledReason: started ? undefined : notStarted, run: a.runReview });
    out.push({ id: "search-tree", group: "analyze", label: "🔬 search tree",
      title: "inspect the MCTS search tree for this combat position",
      disabledReason: inCombat ? undefined : COMBAT_ONLY, run: a.searchTree });
    if (play && !c.staticMode) {
      out.push({ id: "simlab-from-here", group: "analyze", label: "🧪 SimLab from here",
        title: "open this exact position in the SimLab sandbox: edit the hand, piles, enemies and intents, then play it out or compare first moves",
        disabledReason: inCombat ? undefined : COMBAT_ONLY, busy: c.busy.simlabFromHere, run: a.simlabFromHere });
    }
  }
  if ((replay || c.bridgeDirect) && c.bridgeConnected) {
    out.push({ id: "bridge-tree", group: "analyze", label: "🌳 Game tree",
      title: "the LIVE game's run tree — click a save-point node to rewind the real game", run: a.bridgeTree });
    out.push({ id: "live-search", group: "analyze", label: "🔍 Live search",
      title: "MCTS search tree over the live game's current combat board", run: a.liveSearch });
    if (replay && c.viewedFloor != null) {
      out.push({ id: "bridge-seek", group: "analyze", label: `⟲ game → floor ${c.viewedFloor}`,
        title: "rewind the REAL game to the start of the floor you are viewing (the game saves at room entry)",
        busy: c.busy.bridgeSeek, run: a.bridgeSeek });
    }
  }
  return out;
}
