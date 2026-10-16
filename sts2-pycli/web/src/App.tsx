import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import "./theme.css";
import "./game-skin.css";
import "./responsive.css";
import { encodePositionHash } from "./simlab/positionHash";
import { api, DEFAULT_MCTS_CONFIG, NEAR_TIE_DROP_PCT, type Step, type Score, type TimelineEntry, type MapRow, type MapGraph, type FullMap, type LiveTimeline, type SaveMeta, type CombatState, type ResumeCapability, type ImportFidelity, type CapabilityFeatures } from "./api";
import Board from "./components/Board";
import MapView from "./components/MapView";
import MapOverlay from "./components/MapOverlay";
import MiniMap from "./components/MiniMap";
import MapChoices from "./components/MapChoices";
import DecisionScreen from "./components/DecisionScreen";
import AutobattlePanel from "./components/AutobattlePanel";
import GameHud from "./components/GameHud";
import PileViewer from "./components/PileViewer";
import DeckAnalysis from "./components/DeckAnalysis";
import RegretReview from "./components/RegretReview";
import WpaReview from "./components/WpaReview";
import Reasoning from "./components/Reasoning";
import CombatBudget, { type CombatPreset } from "./components/CombatBudget";
import EventLog from "./components/EventLog";
import LiveLog from "./components/LiveLog";
import { activeNode } from "./combatlog";
import BranchTree from "./components/BranchTree";
import SearchTree from "./components/SearchTree";
import Timeline from "./components/Timeline";
import { MctsRuns } from "./components/MctsRuns";
import HelpOverlay from "./components/HelpOverlay";
import { isCombatChoiceStep } from "./choiceKinds";
import SaveLoadDialog from "./components/SaveLoadDialog";
import ImportStateDialog from "./components/ImportStateDialog";
import BridgeConnectDialog from "./components/BridgeConnectDialog";
import { bridgeBase, setBridgeBase } from "./bridge";
import BugReportButton from "./components/BugReportButton";
import BugReportToast, { type BugReportResult } from "./components/BugReportToast";
import BugReportView from "./components/BugReportView";
import PpoPolicyPanel from "./components/PpoPolicyPanel";
import { useShortcuts } from "./useShortcuts";
import { useMediaQuery, NARROW_QUERY, PHONE_LANDSCAPE_QUERY, SHORT_QUERY } from "./useMediaQuery";
import { CollapseButton, PanelGrip } from "./components/panelChrome";
import { usePanelFlag, usePanelHeight } from "./panelPrefs";
import { bestAction, toggleHintsLatch } from "./hint";
import { createActionGate } from "./actionGate";
import { HintsProvider, useHints } from "./hints/HintsContext";
import { isStaticMode, onRestoreProgress } from "./transport";
import { bugReportEndpointAvailable } from "./bugreport";
import OfflineButton from "./components/OfflineButton";
import { BenchButton } from "./components/BenchButton";
import TopBar from "./topbar/TopBar";
import { buildCommands } from "./topbar/commands";
import { usePaletteChord } from "./topbar/CommandPalette";
import TreasureReveal from "./components/TreasureReveal";
import { debugSettingsLine, debugStepLine } from "./debugLog";
import { shouldRerunRegret } from "./regretCache";
import { panelVisible, reasoningView, type LastSearch } from "./searchPanel";
import { liveSeekTargetForDecision } from "./liveSeek";
import { EffectsProvider, useEffectsMode } from "./effects/EffectsContext";
import { SoundProvider, useSound, type SoundValue } from "./sound/SoundContext";
import SoundBridge from "./sound/SoundBridge";
import { cueForUi, type UiCueKind } from "./sound/runCues";
import { actionCues, cardPlayCues } from "./sound/appSoundCues";
import RunEnded from "./components/RunEnded";
import RunLauncher from "./components/RunLauncher";
import { generateSeed } from "./seed";
import { computeRunEnd, shouldShowViewToggle, type EndedRun } from "./runControl";

// Live moves whose parent is a combat decision — the ones the regret coach can
// judge against the MCTS line.
const COMBAT_VERBS = new Set(["play_card", "end_turn", "use_potion"]);

// Rough total floors in a standard 3-act run (each act ≈ 16 floors + boss),
// used only to estimate play-to-end progress / ETA from the current floor.
// The bar is held just shy of 100% until the run actually ends so a longer
// (Act-4 / heart) run doesn't read as "done" early.
const EST_TOTAL_FLOORS = 51;

// Bounds (px) and storage keys for the resizable "machine reasoning" dock. The
// floor is roughly a heading plus one score row: any smaller and the panel is
// worse than collapsed, which is the other button.
const REASONING_MIN_H = 60;
const REASONING_MAX_H = 900;
const REASONING_DEFAULT_H = 260;
const REASONING_COLLAPSED_KEY = "wb.reasoning.collapsed";
const REASONING_HEIGHT_KEY = "wb.reasoning.height";
const AUTOBATTLE_WATCH_KEY = "wb.autobattle.watch";
// Floor on how long one watched combat decision stays on screen. The server side
// of a step is usually slower than this all by itself (an MCTS search takes
// seconds), so it only bites on the cheap steps — an off-combat pick, a fight
// against a fast preset — where without it a whole combat would flash past in
// one frame, which is the opposite of watching it.
const AUTOBATTLE_WATCH_MIN_FRAME_MS = 320;

/** Format a duration in seconds as `m:ss` (or `h:mm:ss` past an hour). */
function fmtDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

/** The hints switch must sit ABOVE everything that reads it, so `App` is a thin
 *  provider shell and the real component is `AppBody`. */
export default function App() {
  return (
    <HintsProvider>
      <AppBody />
    </HintsProvider>
  );
}

/** Stable identity of the chest reveal a state carries, or ``null`` for none.
 *
 *  Used to remember which one you already dismissed. Keyed by the floor plus
 *  the haul, so the next chest re-opens the panel while re-renders of the same
 *  one do not — a plain boolean would either stick shut or flap back open. */
function revealKeyOf(state: CombatState | undefined, scope: string): string | null {
  const r = state?.room_reveal;
  if (!r) return null;
  return `${scope}:${r.kind}:${state?.floor ?? "?"}:${r.gold}:${r.relic?.name ?? "-"}`;
}

/** The connected live game's current position, as state-import text. Module
 *  level so its identity is stable: the import dialog's fetch-on-open effect
 *  depends on it. */
const fetchLiveGameState = () => api.bridge.state().then((s) => JSON.stringify(s));

function AppBody() {
  const [runs, setRuns] = useState<string[]>([]);
  // Follow mode: poll the loaded run file's size/mtime and reload on change,
  // sticking to the newest step — the "watch a live game write its log" view.
  const [loadedPath, setLoadedPath] = useState<string | null>(null);
  const [follow, setFollow] = useState(false);
  const followStatRef = useRef<{ size: number; mtime: number } | null>(null);
  // Did the mod's "Open Workbench" button send us here (`?follow=1`)? That URL
  // promises the LIVE run, so a fallback is worth a visible warning; Follow
  // ticked by hand on an archive is not.
  const followFromUrlRef = useRef(false);
  // Is the followed file the one the ADVISOR says it is writing, or a guess?
  // Without this the checkbox said "· live" either way, so a workbench that
  // had quietly fallen back to the newest file on disk — an unrelated tool's
  // log, when no advisor was running — looked exactly like one tailing the
  // real game (2026-08-22). Only the advisor's own answer earns "live".
  const [followIsLive, setFollowIsLive] = useState(false);
  // Live-GAME bridge (SpireMentor mod via the advisor's control plane).
  const [bridgeConnected, setBridgeConnected] = useState(false);
  // Browser build (GH #709): the control plane URL the page calls directly
  // (src/bridge.ts) and the "Live game…" dialog that sets it.
  const [bridgeUrl, setBridgeUrlState] = useState<string | null>(() => (isStaticMode() ? bridgeBase() : null));
  const [bridgeOpen, setBridgeOpen] = useState(false);
  const [bridgeProbeError, setBridgeProbeError] = useState<string | null>(null);
  const setBridgeUrl = useCallback((url: string | null) => {
    setBridgeBase(url);
    setBridgeUrlState(url);
    setBridgeProbeError(null);
  }, []);
  const [bridgeTimeline, setBridgeTimeline] = useState<import("./api").LiveTimeline | null>(null);
  const [bridgeTreeOpen, setBridgeTreeOpen] = useState(false);
  const [bridgeSeekBusy, setBridgeSeekBusy] = useState(false);
  const [bridgeError, setBridgeError] = useState<string | null>(null);
  const [searchFromBridge, setSearchFromBridge] = useState(false);
  const atTailRef = useRef(true);
  const [count, setCount] = useState(0);
  const [idx, setIdx] = useState(0);
  const [step, setStep] = useState<Step | null>(null);
  const [entries, setEntries] = useState<TimelineEntry[]>([]);
  const [mapRows, setMapRows] = useState<MapRow[]>([]);
  const [mapGraphs, setMapGraphs] = useState<MapGraph[]>([]);

  // Default to the live "play" view so a fresh workbench lands on the Start-run
  // screen rather than a bare "no run loaded". Mount-time live-run rehydration
  // and the `?report=` short-circuit set their own view, so this only governs
  // the empty case.
  const [mode, setMode] = useState<"replay" | "play" | "mcts">("play");
  const [live, setLive] = useState<Step | null>(null);
  // The live sound context, mirrored down from <SoundBridge> (a child of the
  // <SoundProvider> this component renders below — AppBody itself sits
  // OUTSIDE that provider and so can't call useSound() directly). Read by the
  // card-play / map-select / map-open handlers below; SoundBridge itself
  // owns the gold-gain and run-end cues, which are state transitions rather
  // than a single handler call. See sound/SoundBridge.tsx.
  const soundRef = useRef<SoundValue | null>(null);
  /** A UI moment's cue (map open, map select, …) — a no-op while sound is off
   *  or the map hasn't loaded yet. */
  function emitUi(kind: UiCueKind) {
    const sound = soundRef.current;
    if (!sound?.map) return;
    const cue = cueForUi(sound.map, kind);
    if (cue) sound.emit([cue]);
  }
  // A run that was just ended (abandoned mid-run or closed after game over);
  // drives the RunEnded summary shown in the play area until New run / Review
  // replay clears it.
  const [endedRun, setEndedRun] = useState<EndedRun | null>(null);
  // Live event log is the server's branch tree (clickable to rewind/seek).
  const [liveTimeline, setLiveTimeline] = useState<LiveTimeline | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);
  // Whether the LOADED RECORDING can be taken over ("play from here"), from
  // /api/load. Null until a run is loaded. See ResumeCapability.
  const [resume, setResume] = useState<ResumeCapability | null>(null);
  // The node the run is standing on. Its `combat_log` is THIS action's slice —
  // what the board's per-hit effect stream runs on. See `activeNode` for why
  // `live.state.combat_log` is not that (and does not even exist here).
  const liveNode = useMemo(() => activeNode(liveTimeline), [liveTimeline]);
  const [seekBusy, setSeekBusy] = useState(false);
  // Serializes actions applied to the live run. A ref, not state: it must be
  // readable and settable synchronously inside a click/keydown handler, before
  // React has re-rendered anything. See `actionGate` for why a duplicate is
  // dropped rather than queued.
  const actionGate = useRef(createActionGate());
  const [playFromHereBusy, setPlayFromHereBusy] = useState(false);
  const [character, setCharacter] = useState("Ironclad");
  const [seed, setSeed] = useState("");
  const [ascension, setAscension] = useState(0);
  // GH #544: STS2 ships TWO acts at index 0. "roll" reproduces the real
  // lobby's seed-driven pick (uniform 50/50 once Underdocks is unlocked),
  // so it is the default -- anything else shows the wrong act for half of
  // all seeds, which is what made act-1 bug reports irreproducible.
  const [act1, setAct1] = useState("roll");
  const [backend, setBackend] = useState("rust");
  // Engines this deployment can actually run, from /api/capabilities. Starts as
  // rust-only -- the one engine every deployment has -- and grows if the server
  // says C# is spawnable. The list used to be hardcoded, so the deployed
  // (deliberately C#-free) image offered an engine whose pick could only fail.
  const [backends, setBackends] = useState<string[]>(["rust"]);
  // Feature flags from the same probe: a control whose route this deployment
  // cannot answer is hidden rather than left as a button whose every call
  // answers 501. Absent flags (the classic backend) mean everything is on.
  // These are per-FEATURE, never per-mode -- the browser build has grown
  // advisors (M2), run mode (M3), state import and the SimLab sandbox (M5),
  // so gating any of them on `isStaticMode()` now hides something that works.
  const [features, setFeatures] = useState<CapabilityFeatures>({});
  const advisors = features.advisors !== false;
  const runMode = features.run_mode !== false;
  const simlab = features.simlab !== false;
  // The browser build's engine reports `bug_reports: false` (it has no
  // upload target); the HOST may have one — Cloudflare Pages runs the
  // bug-report Function, a plain static host / a local preview do not — so static mode
  // asks the host once and shows the button only where a capture would land.
  const [hostBugReports, setHostBugReports] = useState(false);
  useEffect(() => {
    if (!isStaticMode()) return;
    let cancelled = false;
    void bugReportEndpointAvailable().then((ok) => {
      if (!cancelled) setHostBugReports(ok);
    });
    return () => { cancelled = true; };
  }, []);
  const bugReports = features.bug_reports !== false || hostBugReports;
  const stateImport = features.state_import !== false;
  // "Restoring run n/N": the static build persists the live run in the
  // browser and replays it through the engine on reload, which takes a
  // while for a long run — the worker streams progress frames meanwhile.
  const [restoreProgress, setRestoreProgress] = useState<{ done: number; total: number } | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  // The last bug-report capture, held here rather than in the button: the
  // button is a row in a dropdown that unmounts when it closes.
  const [bugResult, setBugResult] = useState<BugReportResult | null>(null);
  const [bugError, setBugError] = useState<string | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const togglePalette = useCallback(() => setPaletteOpen((o) => !o), []);
  usePaletteChord(togglePalette);
  const [loadOpen, setLoadOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  // Opened via "Import from live game": the dialog fetches on open.
  const [importFromLive, setImportFromLive] = useState(false);
  const [simlabBusy, setSimlabBusy] = useState(false);
  // What the last import could not reproduce; shown above the board until
  // dismissed. Empty for every non-imported run.
  const [importNotes, setImportNotes] = useState<string[]>([]);
  const [saveList, setSaveList] = useState<SaveMeta[]>([]);
  // Name of the save currently being restored, or null. A restore replays the
  // whole saved line through the engine, so a long run takes tens of seconds
  // with nothing to show for it; without this the dialog just sat there
  // looking dead and every further click started ANOTHER concurrent restore
  // (nine of them once, which turned a slow load into a site-wide stall).
  const [loadingSave, setLoadingSave] = useState<string | null>(null);
  const [deckOpen, setDeckOpen] = useState(false);
  // Map overlay — openable from any screen (combat, card reward, shop, …), not
  // just at a `map_select` decision. In play mode the graph is fetched on
  // demand; in replay mode it reuses the run's already-loaded map.
  const [mapOpen, setMapOpen] = useState(false);
  const [mapBusy, setMapBusy] = useState(false);
  const [liveMapGraphs, setLiveMapGraphs] = useState<MapGraph[]>([]);
  // Persistent minimap docked above the side panel (always visible, vs the
  // on-demand overlay). Collapse preference survives reloads.
  const [minimapCollapsed, setMinimapCollapsed] = useState(
    () => localStorage.getItem("minimapCollapsed") === "1",
  );
  // The "machine reasoning" dock at the top of the side panel. A wide search
  // fills it to its cap and leaves the run log a sliver — on a tablet, nearly
  // nothing — so it folds to its header bar and its bottom edge drags. One
  // pair of prefs for both modes: it is the same panel to the person reading
  // it, whether the scores come from the live run or a replayed step.
  const [reasoningCollapsed, setReasoningCollapsed] = usePanelFlag(REASONING_COLLAPSED_KEY, false);
  const [reasoningHeight, setReasoningHeight] =
    usePanelHeight(REASONING_HEIGHT_KEY, REASONING_DEFAULT_H, REASONING_MIN_H, REASONING_MAX_H);
  // Side panel (minimap + run history + reasoning). It is a 320px grid column
  // on a desktop and a drawer over the board on anything narrower, where it
  // would otherwise cost the game a third of its width. `null` = follow the
  // viewport (shown when there is room); pressing the toggle pins a choice, so
  // a tablet user can open it and a desktop user can close it.
  const narrow = useMediaQuery(NARROW_QUERY);
  // Short viewport (a tablet in landscape): in combat the bar between the top
  // row and the HUD holds nothing but "analyze", and 32px is a lot of a 627px
  // screen to spend on one button. It moves into the top row's scroller there.
  const short = useMediaQuery(SHORT_QUERY);
  const combatState = (mode === "play" ? live : step)?.state;
  const analyzeInTopbar = short && mode === "play" && combatState?.decision === "combat_play"
    && (combatState.player?.deck?.length ?? 0) > 0;
  // A phone in landscape (≤520px tall): the top row is a third of the screen
  // the game does not get. It folds away behind the ≡ button and opens OVER
  // the board as a sheet; nothing in it is needed mid-fight.
  const phoneLandscape = useMediaQuery(PHONE_LANDSCAPE_QUERY);
  const [chromeOpen, setChromeOpen] = useState(false);
  const topbarHidden = phoneLandscape && !chromeOpen;
  const [sideOpen, setSideOpen] = useState<boolean | null>(null);
  const sideVisible = sideOpen ?? !narrow;
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [analysis, setAnalysis] = useState<import("./api").DeckAnalysis | null>(null);
  const [regretOpen, setRegretOpen] = useState(false);
  const [regret, setRegret] = useState<import("./api").RegretReview | null>(null);
  // Height (px) of the fixed regret dock, reported by the component. Scroll
  // regions reserve this much bottom padding so the dock can't hide content
  // (the card row, the tail of the history log) with no way to scroll to it.
  const [regretHeight, setRegretHeight] = useState(0);
  // Which line the cached `regret` analysis is for (true = live run, false =
  // loaded replay). Lets the dock reuse the last analysis instead of re-running
  // the expensive diff every time it's reopened. Reset to null to invalidate.
  const regretLiveRef = useRef<boolean | null>(null);
  // WPA run review (issue #201) — the run-level twin of the regret dock:
  // outlook curve + turning points over the map/card/shop/rest decisions.
  // Same caching scheme as `regret`; mutually exclusive with the regret dock
  // (both are fixed bottom overlays).
  const [wpaOpen, setWpaOpen] = useState(false);
  const [wpa, setWpa] = useState<import("./api").WpaReview | null>(null);
  const [wpaHeight, setWpaHeight] = useState(0);
  const wpaLiveRef = useRef<boolean | null>(null);
  // Autobattle mode: the run-level drill. Combats are auto-played by the MCTS
  // agent so only run-level decisions reach the human, with a change-diff at
  // each stop and the champion's pick revealed AFTER the pick is committed.
  const [autobattleOn, setAutobattleOn] = useState(false);
  const [autobattleDiff, setAutobattleDiff] = useState<import("./api").AutobattleDiff | null>(null);
  const [autobattleVerdict, setAutobattleVerdict] =
    useState<import("./api").AutobattleVerdict | null>(null);
  const [autobattleBusy, setAutobattleBusy] = useState(false);
  const [autobattleOver, setAutobattleOver] = useState({ over: false, won: false });
  // Watch the auto-played fights instead of skipping them: the leg is walked one
  // decision per request and painted, so the combats the agent plays are on
  // screen. Off = the whole leg resolves in one server-side call (much faster on
  // a long fight; the diff still summarizes what it cost). Persisted because it
  // is a taste setting, not a per-run one.
  const [autobattleWatch, setAutobattleWatch] = usePanelFlag(AUTOBATTLE_WATCH_KEY, true);
  // Set while a watched leg is being walked, so Esc can interrupt it (the loop
  // body checks `stopAutoRef` between steps) without borrowing `autoRunning`,
  // whose progress overlay and reduced-effects mode belong to "play to end".
  const autobattleWatchRef = useRef(false);
  // Live regret coach: after each combat move, flag it if it was a blunder.
  const [coachOn, setCoachOn] = useState(false);
  const [blunder, setBlunder] = useState<import("./api").DecisionRegret | null>(null);
  const [trend, setTrend] = useState<import("./api").TrendPoint[]>([]);
  const [treeOpen, setTreeOpen] = useState(false);
  const [searchTree, setSearchTree] = useState<import("./api").SearchTree | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchConfig, setSearchConfig] = useState<import("./api").MctsConfig>(
    () => ({ ...DEFAULT_MCTS_CONFIG }),
  );
  const [searchBusy, setSearchBusy] = useState(false);
  // MCTS knobs for the hints we show (H / play-best / autoplay). Same defaults
  // as the search-tree inspector's, so the two agree until you change one.
  const [hintConfig, setHintConfig] = useState<import("./api").MctsConfig>(
    () => ({ ...DEFAULT_MCTS_CONFIG }),
  );
  const hintConfigRef = useRef(hintConfig);
  hintConfigRef.current = hintConfig;
  const [liveAttack, setLiveAttack] = useState<{ target: number; nonce: number } | null>(null);
  // One switch for every AI opinion in the app — combat hints included. Default
  // OFF: a human plays the game, and asking what the AI thinks is deliberate.
  const hints = useHints();
  const autoHints = hints.on && advisors;
  const [dismissedReveal, setDismissedReveal] = useState<string | null>(null);
  const [hintBusy, setHintBusy] = useState(false);
  const [recommendBusy, setRecommendBusy] = useState(false);
  // Pressing H asks for the machine's read of the position in front of you — a
  // deliberate act, so it works with the global switch off.
  // This latch is what makes "you asked" distinguishable from "it just
  // appeared": it is set by liveHint and cleared by any action or seek, so the
  // scores are shown for the position you asked about and no further. Autoplay
  // deliberately does NOT set it — handing the fight to the AI shouldn't leave
  // the answer on screen for the next move you make yourself.
  const [hintShown, setHintShown] = useState(false);
  // (`showScores`, which gates the overlay + the reasoning panel, is derived
  // further down — it also reads `autoRunning`/`autoPaused`/`lastSearch`.)
  const [autoRunning, setAutoRunning] = useState(false); // an autoplay (A) loop is active
  // "Play to end" combat search budgets (the unified stepper always uses the
  // combat MCTS agent, regardless of the off-combat brain). Two knobs:
  //  • monster sims — MCTS iters in regular (non-boss) combat (`sims`)
  //  • boss sims    — MCTS iters in boss combat (`boss_sims`)
  // Defaults are the "Benchmark" preset (1000/4000) — the offline win-rate
  // sweep's budget, so a run here is comparable to a benchmarked number by
  // default. "Quick" (200/800) is the opt-in for fast iteration.
  //
  // A third knob, `reward_sims`, used to budget a boss-sim rollout per
  // card-reward candidate. It is gone: the off-combat brain picks rewards now,
  // and the server documents the field as "accepted but ignored" (see
  // StartFaithfulRequest). Don't put the input back without a reader.
  const [monsterSims, setMonsterSims] = useState(1000);
  const [bossSims, setBossSims] = useState(4000);
  // Root-parallel MCTS workers for the parity ("play to end") run.
  //
  // NOT WIRED on this path, which is why it defaults to 1: the server hands it
  // to `sts2_rs.HeuristicRunAgent`, whose constructor discards it
  // (`heuristic_agent.rs`) — combat search here is always single-threaded. The
  // recommend-panel `threads` knob (MctsControls -> mcts_recommend) is a
  // different code path and does honour it.
  //
  // It stays visible, disabled and labelled, rather than being deleted: the
  // knob reads as a live setting in screenshots and old notes, and a box that
  // is present-but-explicitly-inert is a clearer answer to "did I run that at
  // 6 threads?" than no box at all.
  const [faithfulThreads] = useState(1);
  // Optional debug log: a running trace of what the autoplay loop is doing
  // (path resolution, combat MCTS sim budgets, card-reward boss sims, …).
  const [debugEnabled, setDebugEnabled] = useState(false);
  const debugEnabledRef = useRef(false);
  debugEnabledRef.current = debugEnabled;
  const [debugLog, setDebugLog] = useState<string[]>([]);
  // Collapsed = drawer header only (fixed height), expanded = header + fixed
  // scroll body. Both are a fixed height so toggling never shifts the board.
  const [debugCollapsed, setDebugCollapsed] = useState(false);
  // Append one timestamped line to the debug log (bounded), no-op when the log
  // is off. Cheap enough to call every step.
  const pushDebug = (line: string) => {
    if (!debugEnabledRef.current) return;
    const t = new Date().toLocaleTimeString(undefined, { hour12: false });
    setDebugLog((l) => {
      const next = [...l, `${t}  ${line}`];
      return next.length > 1000 ? next.slice(next.length - 1000) : next;
    });
  };
  // Two-axis run mode: off-combat brain (which policy drives card rewards /
  // shops / events / map / rest) x combat search preset (sim budget for the
  // combat MCTS agent, which always runs regardless of brain). Both
  // axes are posted server-side via `api.live.runMode` and then driven move-
  // by-move by the single unified `api.live.runModeStep` stepper.
  const [offCombatBrain, setOffCombatBrain] = useState<"heuristic" | "ppo">("ppo");
  const [combatPreset, setCombatPreset] = useState<CombatPreset>("benchmark");
  // The budget the SERVER is searching at, as last posted. Kept apart from the
  // three inputs above so the live toolbar can tell "you typed 4000" from "the
  // run is searching at 4000" — without it there is no way to see whether a
  // mid-run change landed. Null until a run-mode run starts.
  const [appliedBudget, setAppliedBudget] = useState<{ monster: number; boss: number } | null>(null);
  // Root distribution of the last combat search the run-mode stepper ran, so
  // the reasoning sidebar can show an autoplayed move's branches. See
  // searchPanel.ts for why these never merge into `live.scores`.
  const [lastSearch, setLastSearch] = useState<LastSearch | null>(null);
  // One-line summary of the run settings for the debug log header: engine,
  // brain, combat search budget, and the per-decision sim budgets. Emitted at
  // run start and whenever the drawer is toggled on mid-run.
  const settingsLine = () =>
    debugSettingsLine({
      character, ascension, seed, backend, brain: offCombatBrain,
      combat: combatPreset, threads: faithfulThreads, monsterSims, bossSims,
    });
  // Toolbar 🐞 toggle: show/hide the debug drawer at any time (not just before a
  // run). Turning it on mid-run re-emits the settings header so the drawer opens
  // with context. The ref is set synchronously so the immediate pushDebug fires.
  const toggleDebug = () => {
    const next = !debugEnabledRef.current;
    debugEnabledRef.current = next;
    setDebugEnabled(next);
    if (next) pushDebug(settingsLine());
  };
  const [autoPaused, setAutoPaused] = useState(false); // Stop pressed, run resumable
  // Board score overlay + reasoning panel. Advice is opt-in, so this is NOT
  // simply `hints.on`; see `panelVisible` for the two carve-outs (you pressed
  // H, or the AI is playing the run and there is no move of yours to spoil).
  const showScores = panelVisible({
    hintsOn: hints.on, hintShown, autoRunning, autoPaused, lastSearch,
  });
  // Live progress for the play-to-end bar. A single run-mode step can block for
  // seconds (boss-sim card rewards, combat MCTS) inside one request, so the
  // screen would otherwise freeze with no feedback. `autoStepStartRef` is reset
  // at the top of every loop iteration; the elapsed readout ticks off it.
  const [autoStepNum, setAutoStepNum] = useState(0);
  const [autoStepElapsed, setAutoStepElapsed] = useState(0);
  const autoStepStartRef = useRef(0);
  // Whole-run progress for the play-to-end bar: total elapsed (ticked off
  // `autoRunStartRef`) plus the run's current floor/act, used to estimate %
  // complete and an ETA from how far through the run we are.
  const [autoRunElapsed, setAutoRunElapsed] = useState(0);
  const autoRunStartRef = useRef(0);
  const [autoFloor, setAutoFloor] = useState<number | null>(null);
  const [autoAct, setAutoAct] = useState<number | null>(null);
  const autoRunningRef = useRef(false); // mirror of autoRunning for the global stop listener
  const stopAutoRef = useRef(false); // set to interrupt an in-progress autoplay loop (Esc)
  const hintInFlight = useRef(false); // a hint HTTP call is in progress
  const stepHealRef = useRef(false); // a session-recovery reload is in progress
  const autoHintedRef = useRef(false); // already auto-fetched for the CURRENT decision

  /** Re-scan the logs dir for run files. Called on mount AND after a run ends:
   *  the list used to be fetched only at mount, so the log the run you just
   *  finished wrote was missing from the "load a run…" picker until a reload. */
  const refreshRunList = useCallback(async () => {
    try {
      const r = await api.runs();
      setRuns(r.runs);
    } catch {
      /* listing runs is best-effort; a failure must not break the run flow */
    }
  }, []);
  useEffect(() => { void refreshRunList(); }, [refreshRunList]);
  // `?follow=1[&run=<path>]` (the mod's "Open Workbench" button): open the
  // named run — or the newest one — in the replay view with Follow on.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("follow") !== "1") return;
    followFromUrlRef.current = true;
    let cancelled = false;
    (async () => {
      let path = params.get("run");
      // The live advisor knows exactly which file it is writing — prefer that
      // over the /api/runs head, which is only ever a guess at which file the
      // game is in.
      const live = await api.bridge.status()
        .then((s) => s.log_path ?? null)
        .catch(() => null);
      if (!path) path = live;
      if (!path) {
        const r = await api.runs();
        path = r.runs[0] ?? null;
      }
      if (!path || cancelled) return;
      setMode("replay");
      const n = await loadRun(path);
      if (cancelled) return;
      setFollow(true);
      setFollowIsLive(!!live && live === path);
      // A live view opens at the newest step, not the run's first.
      if (n > 0) setIdx(n - 1);
    })().catch(() => {});
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // Probe the live-game bridge while the replay view is up: connected means
  // a real game + advisor are running, which unlocks the Game-tree / live
  // search buttons. Cheap (503 when the advisor is down), every 5 s.
  // In the browser build the probe runs whenever a control plane URL is
  // configured, in any view — the live-game views are not tied to the
  // replay of an advisor log there (GH #709).
  useEffect(() => {
    if (mode !== "replay" && !bridgeUrl) { setBridgeConnected(false); return; }
    let cancelled = false;
    const probe = () =>
      api.bridge.status()
        .then((s) => { if (!cancelled) { setBridgeConnected(!!s.connected); setBridgeProbeError(null); } })
        .catch((e: unknown) => {
          if (cancelled) return;
          setBridgeConnected(false);
          if (bridgeUrl) setBridgeProbeError(e instanceof Error ? e.message : String(e));
        });
    probe();
    const id = setInterval(probe, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, [mode, bridgeUrl]);
  // Stick-to-tail bookkeeping: following only auto-advances when the user was
  // already looking at the newest step (scrubbing back pauses the auto-seek;
  // returning to the tail resumes it).
  useEffect(() => { atTailRef.current = count === 0 || idx >= count - 1; }, [idx, count]);
  // The Follow poll: a cheap size/mtime probe every 2 s; only a change pays
  // for a full /api/load + timeline refresh.
  useEffect(() => {
    if (!follow || mode !== "replay") return;
    let cancelled = false;
    const id = setInterval(async () => {
      try {
        // An advisor restart (coach toggled off/on, next game session)
        // writes a NEW log file — adopt it, or this poll would stat the
        // dead previous file forever ("workbench lost track", 2026-08-21).
        const live = await api.bridge.status()
          .then((s) => s.log_path ?? null)
          .catch(() => null);
        if (cancelled) return;
        if (live && live !== loadedPath) {
          const n = await loadRun(live);
          if (cancelled) return;
          setFollowIsLive(true); // after loadRun, which clears it
          if (n > 0) setIdx(n - 1);
          return;
        }
        // Re-decided every tick, in both directions: an advisor that comes up
        // mid-session promotes a guessed file to a live one, and an advisor
        // that goes away demotes it back.
        setFollowIsLive(!!live);
        if (!loadedPath) return;
        const st = await api.runStat(loadedPath);
        const prev = followStatRef.current;
        followStatRef.current = st;
        if (cancelled || (prev && prev.size === st.size && prev.mtime === st.mtime)) return;
        const { count: n } = await api.load(loadedPath);
        const { steps } = await api.timeline();
        if (cancelled) return;
        setEntries(steps);
        setCount(n);
        if (atTailRef.current && n > 0) setIdx(n - 1);
      } catch {
        /* transient poll errors are fine; the next tick retries */
      }
    }, 2000);
    return () => { cancelled = true; clearInterval(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [follow, loadedPath, mode]);
  // Ask the server which engines it can spawn, so the dropdown never offers a
  // dead one. On a failed probe we keep the rust-only default rather than
  // guessing wider: a missing engine is the failure this whole path exists to
  // prevent. Runs once on mount.
  useEffect(() => {
    let cancelled = false;
    api
      .capabilities()
      .then(({ backends: available, features: flags }) => {
        if (cancelled) return;
        if (flags) setFeatures(flags);
        if (!available?.length) return;
        // The browser-only build has exactly one engine, whatever it reports.
        const list = isStaticMode() ? ["rust"] : available;
        setBackends(list);
        // A backend pinned before the probe answered (or carried over from an
        // older build) must not survive as an unrunnable selection.
        setBackend((b) => (list.includes(b) ? b : list[0]));
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);
  // Restore an in-progress live run on page load. The backend keeps the live
  // session alive across page reloads, but the React app boots empty — so
  // without this, closing/reopening (or navigating away and back) strands the
  // run on a blank workbench. If a live run exists, rehydrate it and switch to
  // the play view; a 400 ("no live run") just leaves us on the default play
  // view's Start screen. Runs once on mount.
  useEffect(() => {
    let cancelled = false;
    // Static build only: the worker reports how far it is through replaying
    // the persisted run (a no-op subscription against the Python backend).
    const offProgress = onRestoreProgress((p) => { if (!cancelled) setRestoreProgress(p); });
    api.live
      .state()
      .then((s) => {
        if (cancelled) return;
        setLive(s);
        setMode("play");
        void refreshTimeline();
      })
      .catch(() => {}) // no active run → stay on the play view's Start screen
      .finally(() => { offProgress(); if (!cancelled) setRestoreProgress(null); });
    return () => { cancelled = true; offProgress(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // Tick the play-to-end step timer while a run is active. The loop can't update
  // state mid-await, so the current step's elapsed seconds come from an interval
  // reading `autoStepStartRef` (reset at the top of each loop iteration).
  useEffect(() => {
    if (!autoRunning) { setAutoStepElapsed(0); return; }
    const id = setInterval(() => {
      setAutoStepElapsed((Date.now() - autoStepStartRef.current) / 1000);
      setAutoRunElapsed((Date.now() - autoRunStartRef.current) / 1000);
    }, 100);
    return () => clearInterval(id);
  }, [autoRunning]);
  // Esc interrupts an in-progress autoplay / play-to-end loop, from any screen
  // (the per-loop body checks `stopAutoRef` each step). Registered once.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape" && (autoRunningRef.current || autobattleWatchRef.current))
        stopAutoRef.current = true;
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    if (count <= 0) return;
    let cancelled = false;
    api.step(idx)
      .then((s) => { if (!cancelled) setStep(s); })
      .catch(async () => {
        // A workbench-server restart wipes the in-memory session while this
        // page still shows a loaded run; re-load it once and retry.
        if (cancelled || !loadedPath || stepHealRef.current) return;
        stepHealRef.current = true;
        try {
          await api.load(loadedPath);
          const s = await api.step(idx);
          if (!cancelled) setStep(s);
        } catch {
          /* server really down; the follow poll retries on its own */
        } finally {
          stepHealRef.current = false;
        }
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx, count]);
  useEffect(() => {
    if (!autoHints || !live || live.decision !== "combat_play" || autoRunning) return;
    if (live.scores.length > 0 || hintInFlight.current || autoHintedRef.current) return;
    autoHintedRef.current = true; // latch BEFORE fetching so an empty result can't loop
    fetchHint();
    // fetchHint replaces `live` with a fresh step (with scores), re-running this
    // effect; the scores-length / latch guards then stop it. The latch is cleared
    // by the next action (Step 3b) or after autoplay, which re-enables a fetch.
    // ``autoRunning`` is a dep so the effect re-fires (and fetches a fresh hint)
    // once an autoplay press finishes, rather than leaving the panel on
    // "thinking…" indefinitely.
  }, [autoHints, live, autoRunning]);

  // Re-run the hint when the MCTS knobs change, so editing sims/depth/etc.
  // refreshes the displayed search instead of leaving stale values. Only fires
  // when a hint is already active (auto-hints on, or scores currently shown);
  // debounced so typing a multi-digit value doesn't spam the search.
  useEffect(() => {
    const inCombat = live?.decision === "combat_play";
    const active = autoHints || (live?.scores.length ?? 0) > 0;
    if (!inCombat || !active || autoRunning) return;
    const t = setTimeout(() => {
      autoHintedRef.current = true; // suppress the auto-fetch effect's own fire
      fetchHint();
    }, 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hintConfig]);

  useEffect(() => { localStorage.setItem("minimapCollapsed", minimapCollapsed ? "1" : "0"); }, [minimapCollapsed]);

  // Keep the persistent minimap fresh in play mode: refetch the current act's
  // graph whenever the live run advances to a new floor/act (the overlay's
  // on-demand fetch only fires when opened, so without this the docked minimap
  // would lag behind the run). Replay reuses the run's already-loaded graphs.
  useEffect(() => {
    if (mode !== "play" || !live) return;
    let cancelled = false;
    api.live.map()
      .then(({ graphs }) => { if (!cancelled) setLiveMapGraphs(graphs); })
      .catch(() => {});
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, live?.meta.act, live?.meta.floor, !!live]);

  async function loadRun(path: string): Promise<number> {
    const { count: n, resume: cap } = await api.load(path);
    setResume(cap ?? null);
    setLiveError(null);
    const [{ steps }, { rows, graphs }] = await Promise.all([api.timeline(), api.map()]);
    setEntries(steps);
    setMapRows(rows);
    setMapGraphs(graphs);
    setCount(n);
    setIdx(0);
    setLoadedPath(path);
    followStatRef.current = null;
    // Any file loaded here is a guess until something proves otherwise; the
    // two callers that DO know it is the advisor's set the flag back after.
    setFollowIsLive(false);
    // A fresh file (written in the last few minutes) is probably a live game —
    // default Follow on for it, off for cold archives.
    api.runStat(path)
      .then((st) => setFollow(Date.now() / 1000 - st.mtime < 300))
      .catch(() => setFollow(false));
    // New run loaded — any cached regret analysis is for the old line. These
    // two sat AFTER the return, so they never ran and a stale review carried
    // across every load.
    setRegret(null);
    regretLiveRef.current = null;
    return n;
  }
  /** Open a recorded MCTS run (by filename) in the timeline replay. The runs
   *  table reports a bare filename; the /api/runs list holds full paths, so
   *  resolve the matching full path (falling back to the filename) before
   *  loading, then switch to the replay view. */
  async function openMctsRun(path: string) {
    const full = runs.find((r) => r === path || r.split("/").pop() === path) ?? path;
    setMode("replay");
    await loadRun(full);
  }
  async function recommend() {
    if (recommendBusy) return;
    setRecommendBusy(true);
    setBridgeError(null);
    try {
      const r = await api.recommend(idx, { sims: 2000 });
      setStep((s) => (s ? { ...s, scores: r.scores, pv: r.pv } : s));
    } catch (e) {
      // Surfaced instead of swallowed: a dead session or backend error made
      // the button read as "does nothing" (live report 2026-08-21).
      setBridgeError(e instanceof Error ? e.message : String(e));
    } finally {
      setRecommendBusy(false);
    }
  }

  /** Open the deck-analysis modal, grading the current deck (live or replay).
   *  In replay it also pulls the run-wide axis trend for the sparklines. */
  async function openAnalysis() {
    setAnalysis(null);
    setTrend([]);
    setAnalysisOpen(true);
    try {
      const r = mode === "play" ? await api.live.deckAnalysis() : await api.deckAnalysis(idx);
      setAnalysis(r);
    } catch {
      setAnalysis({ empty: true } as import("./api").DeckAnalysis);
    }
    if (mode === "replay") {
      api.deckTrend().then((t) => setTrend(t.points)).catch(() => setTrend([]));
    }
  }

  /** Open the map overlay. Replay reuses the run's already-loaded map; play
   *  fetches the current act's graph on demand (works at any decision). */
  async function openMap() {
    setMapOpen(true);
    if (mode !== "play") return;
    emitUi("mapOpen");
    setMapBusy(true);
    try {
      const { graphs } = await api.live.map();
      setLiveMapGraphs(graphs);
    } catch {
      setLiveMapGraphs([]);
    } finally {
      setMapBusy(false);
    }
  }

  /** Run the post-combat regret analysis: diff every recorded decision against
   *  the MCTS line and surface the costliest plays (the learning engine).
   *  `live` reviews the in-progress run's active line instead of a loaded
   *  replay (combat positions are re-scored on demand server-side). The result
   *  is cached on `regret`; `regretLiveRef` records which line it's for. */
  async function runRegret(live = false) {
    regretLiveRef.current = live;
    setRegret(null);
    try {
      setRegret(await (live ? api.live.regretReview({}) : api.regretReview({})));
    } catch {
      setRegret({
        decisions: [], blunders: [],
        summary: { total_decisions: 0, blunder_count: 0, cumulative_drop_pct: 0, worst_drop_pct: 0 },
      });
    }
  }

  /** Toggle the regret-review dock. The analysis is expensive, so reopening
   *  reuses the cached result (only re-running when there's none yet or the
   *  cached line no longer matches the current mode). Use the dock's ↻ button
   *  to force a fresh diff (e.g. after more live moves). */
  function openRegret(live = false) {
    if (regretOpen) { setRegretOpen(false); return; }
    setWpaOpen(false); // one bottom dock at a time
    setRegretOpen(true);
    if (shouldRerunRegret(regret != null, regretLiveRef.current, live)) void runRegret(live);
  }

  /** Fetch the WPA run review — the run-level value curve + turning points
   *  (issue #201). `live` reviews the in-progress run's active line; otherwise
   *  the loaded replay's recorded values. Cheap server-side (values were
   *  recorded during play), but cached like the regret review for symmetry. */
  async function runWpa(live = false) {
    wpaLiveRef.current = live;
    setWpa(null);
    try {
      setWpa(await (live ? api.live.wpaReview({}) : api.wpaReview({})));
    } catch {
      setWpa({
        available: false, reason: "review failed — is a run loaded?",
        points: [], turning_points: [], summary: {}, caveats: [],
      });
    }
  }

  /** Toggle the WPA run-review dock (mutually exclusive with the regret dock —
   *  both pin to the bottom of the viewport). */
  function openWpa(live = false) {
    if (wpaOpen) { setWpaOpen(false); return; }
    setRegretOpen(false);
    setWpaOpen(true);
    if (shouldRerunRegret(wpa != null, wpaLiveRef.current, live)) void runWpa(live);
  }

  async function refreshTimeline() {
    try {
      setLiveTimeline(await api.live.timeline());
    } catch {
      setLiveTimeline(null);
    }
  }
  /** Drop all replay-viewer state so a fresh live run never inherits the
   *  previously-loaded run's steps (e.g. the bottom Timeline, which renders
   *  whenever `count > 0`, was still showing the old replay's floors). */
  function resetReplayView() {
    setEntries([]);
    setCount(0);
    setIdx(0);
    setMapRows([]);
    setMapGraphs([]);
  }
  async function startLive() {
    autoHintedRef.current = false;
    setLiveError(null);
    setEndedRun(null);
    resetReplayView();
    // A rejected start MUST land in the error banner. Unhandled, it read as
    // "Start run does nothing" -- which is exactly how the deployed site
    // behaved when the C# engine was picked and the server had no .NET to
    // spawn (the 500 went straight to an unhandled promise rejection).
    // An empty seed field means "roll one now" — and show it, so the run on
    // screen is the run the seed names (the launcher itself pre-rolls one).
    const runSeed = seed || generateSeed();
    if (runSeed !== seed) setSeed(runSeed);
    let s: Step;
    try {
      s = await api.live.start({ character, seed: runSeed, ascension, backend, act1 });
    } catch (e) {
      setLiveError(`start failed: ${e}`);
      return;
    }
    setLive(s);
    setAutoPaused(false);
    setLastSearch(null);
    setAppliedBudget(null);
    // Fresh run — drop any cached regret analysis from a previous line.
    setRegret(null);
    regretLiveRef.current = null;
    await refreshTimeline();
  }
  /** Whether "play from here" can take over the VIEWED step of the loaded
   *  recording. A fixture-sourced combat recording can only be resumed at a
   *  combat decision; a workbench run log rebuilds from its seed, so any step
   *  (shop, card reward, map, mid-combat) is fair game. */
  function canPlayFromHere(): boolean {
    if (mode !== "replay" || !resume) return false;
    if (resume.kind === "run") return count > 0;
    if (resume.kind === "combat") return step?.decision === "combat_play";
    return false;
  }
  /** Take over the loaded recording at the viewed step, interactively (branch
   *  tree + hints): a combat fixture reloads its fixture, a workbench run log
   *  rebuilds the whole run from its seed. */
  async function playFromHere() {
    if (playFromHereBusy || !canPlayFromHere()) return;
    autoHintedRef.current = false;
    setLiveError(null);
    setEndedRun(null);
    // Rebuilding a full run replays every recorded action through the engine —
    // seconds on a deep run, during which the button used to look inert.
    setPlayFromHereBusy(true);
    try {
      const s = await api.live.fromReplay(idx);
      setMode("play");
      setLive(s);
      await refreshTimeline();
    } catch (e) {
      setLiveError(`play-from-here failed: ${e}`);
    } finally {
      setPlayFromHereBusy(false);
    }
  }
  async function endRun() {
    if (!live) return;
    if (!window.confirm("End this run? The in-progress game will be closed.")) return;
    // Snapshot for the summary, then end WITHOUT letting a quit() failure strand
    // the run on screen: computeRunEnd swallows the reject (e.g. 400 "no live
    // run" when the run already reached game over) so the teardown below always
    // runs. Land on the RunEnded summary in the play area.
    const ended = await computeRunEnd(live, () => api.live.quit());
    setLive(null);
    setAutoPaused(false);
    setLiveTimeline(null);
    setLiveError(null);
    setTreeOpen(false);
    setEndedRun(ended);
    setMode("play");
    await refreshRunList();
  }
  /** Live game-over → "Review replay": close the finished run (which writes
   *  its log) and open that log in the replay viewer. Without this the only
   *  route to a replay was the toolbar's red "End run", which players reaching
   *  a natural game over never think to press. */
  async function reviewFinishedRun() {
    if (!live) return;
    const ended = await computeRunEnd(live, () => api.live.quit());
    setLive(null);
    setAutoPaused(false);
    setLiveTimeline(null);
    setTreeOpen(false);
    await refreshRunList();
    if (!ended.logPath) {
      // Quit failed or the run left no log: land on the summary rather than a
      // blank replay view, and say so instead of failing silently.
      setEndedRun(ended);
      setMode("play");
      setLiveError("this run left no recorded log to replay");
      return;
    }
    setLiveError(null);
    setMode("replay");
    await loadRun(ended.logPath);
  }
  /** RunEnded → "New run": clear the summary and fall through to the Start
   *  screen (same character pre-selected). */
  function startNewRunFromSummary() {
    setEndedRun(null);
  }
  /** RunEnded → "Review replay": open the just-finished run in the replay
   *  viewer. Only reachable when the ended run left a recorded log. */
  async function reviewEndedReplay() {
    const path = endedRun?.logPath;
    setEndedRun(null);
    if (!path) return;
    setMode("replay");
    await loadRun(path);
  }
  async function saveLive() {
    if (!live) return;
    const name = window.prompt("Save name:");
    if (!name || !name.trim()) return;
    try {
      const r = await api.live.save(name.trim());
      setLiveError(null);
      window.alert(`Saved “${r.name}” (${r.step_count} steps).`);
    } catch (e) {
      setLiveError(`save failed: ${e}`);
    }
  }
  async function openLoadDialog() {
    try {
      const { saves } = await api.saves.list();
      setSaveList(saves);
      setLoadOpen(true);
    } catch (e) {
      setLiveError(`could not list saves: ${e}`);
    }
  }
  async function loadSaved(name: string) {
    // The guard lives here rather than in the dialog so it holds however the
    // click arrived — a restore cannot be cancelled server-side, so a second
    // one only competes with the first for the same CPU.
    if (loadingSave !== null) return;
    autoHintedRef.current = false;
    setLiveError(null);
    setEndedRun(null);
    setLoadingSave(name);
    try {
      const s = await api.saves.load(name);
      setMode("play");
      setLive(s);
      setLoadOpen(false);
      await refreshTimeline();
    } catch (e) {
      setLoadOpen(false);
      setLiveError(`load failed: ${e}`);
    } finally {
      setLoadingSave(null);
    }
  }
  /** Adopt a just-imported live state as the active run.
   *
   * The dialog has already done the import; this only re-points the UI at it.
   * `setLive` from the import response would do, but the timeline has to be
   * refetched anyway (the branch tree is now rooted at the imported node), and
   * re-reading the state keeps one source of truth for what is on screen.
   */
  async function afterImport(fidelity: ImportFidelity) {
    autoHintedRef.current = false;
    setLiveError(null);
    setEndedRun(null);
    setImportOpen(false);
    // Carried onto the board, not left in the dialog: these are what the
    // rebuilt position does NOT reproduce, and the whole point is that a
    // half-restored state must not read as a whole one.
    setImportNotes(fidelity.warnings ?? []);
    try {
      setMode("play");
      setLive(await api.live.state());
      await refreshTimeline();
    } catch (e) {
      setLiveError(`import failed: ${e}`);
    }
  }
  /** Open SimLab's sandbox on the current board.
   *
   * The position rides in the URL hash (`#p=…`), so the new tab starts on the
   * exact hand, pile order, powers and telegraphs — and nothing is queued
   * until the user presses Run there. A new tab rather than navigation: the
   * analysis takes a while, and losing the position you launched it from to
   * look at it would defeat the point.
   */
  async function simlabFromHere() {
    setSimlabBusy(true);
    try {
      const r = await api.live.position();
      const url = `/simlab.html#p=${encodePositionHash(r.position)}`;
      window.open(url, "_blank", "noopener");
    } catch (e) {
      setLiveError(`SimLab from here failed: ${e}`);
    } finally {
      setSimlabBusy(false);
    }
  }
  /** Auto-play up to the next run-level decision (autobattle mode).
   *
   *  Two paths over the same leg: WATCH walks it one decision per request and
   *  paints each one, so the fights the agent plays are actually on screen;
   *  SKIP resolves the whole leg in a single server-side call and shows only
   *  where it landed. The stop→stop change-diff spans the leg either way, so
   *  switching mid-run (or pressing Esc out of a watch) stays honest. */
  async function advanceAutobattle() {
    if (autobattleBusy) return;
    setAutobattleBusy(true);
    // The loop paints every step itself; letting the auto-hint effect fire on
    // each one would queue a search per painted frame.
    autoHintedRef.current = true;
    try {
      if (autobattleWatch) await watchAutobattleLeg();
      else await skipAutobattleLeg();
    } catch (e) {
      setLiveError(`autobattle failed: ${e}`);
    } finally {
      autoHintedRef.current = false;
      setAutobattleBusy(false);
      await refreshTimeline();
    }
  }
  /** Skip path: the whole leg resolves in one call, fights unseen. */
  async function skipAutobattleLeg() {
    const r = await api.live.autobattleAdvance();
    setLive(r.step);
    setAutobattleDiff(r.diff);
    setAutobattleOver({ over: r.over, won: r.won });
    setLiveError(r.error ?? null);
  }
  /** Watch path: one painted decision per request until the next stop.
   *
   *  Esc (or the panel's Stop) leaves the run exactly where it stands rather
   *  than unwinding — pressing the panel's button again resumes the same leg,
   *  because the diff window lives on the server and is still open. */
  async function watchAutobattleLeg() {
    autobattleWatchRef.current = true;
    stopAutoRef.current = false;
    // A full timeline fetch per painted step is O(N) fetches of a growing
    // payload over a long fight; the `finally` in advanceAutobattle always does
    // a last exact one. Same throttle as the play-to-end loop.
    const TIMELINE_STRIDE = 6;
    let sinceRefresh = TIMELINE_STRIDE; // refresh on the first step, for liveness
    try {
      for (let i = 0; i < 5000; i++) {
        const started = Date.now();
        const r = await api.live.autobattleStep();
        setLive(r.step);
        // Off-combat steps carry no search; keep the last combat one on screen
        // rather than blanking the panel between fights.
        if (r.search) setLastSearch(r.search);
        if (r.debug?.summary) pushDebug(debugStepLine(r.debug));
        if (r.diff) setAutobattleDiff(r.diff);
        setAutobattleOver({ over: r.over, won: r.won });
        setLiveError(r.error ?? null);
        if (++sinceRefresh >= TIMELINE_STRIDE) {
          sinceRefresh = 0;
          await refreshTimeline();
        }
        // `played` false means the run was already sitting on a stop: the guard
        // that keeps a non-advancing state from spinning the loop.
        if (r.stopped || r.over || r.error || !r.played) break;
        if (stopAutoRef.current) break;
        const left = AUTOBATTLE_WATCH_MIN_FRAME_MS - (Date.now() - started);
        if (left > 0) await new Promise((done) => setTimeout(done, left));
      }
    } finally {
      autobattleWatchRef.current = false;
      stopAutoRef.current = false;
    }
  }
  /** Apply one action to the live run, dropping it if another is already in
   *  flight. Returns the resulting step (null on an engine error, and null on a
   *  drop) so a caller driving a multi-step decision can look at where it landed
   *  without racing the `live` state update.
   *
   *  The gate is the fix for bug wb-12ce3fe830de0bb1: until this returns, the
   *  screen on display is the one the action is LEAVING, so every affordance on
   *  it — a map node, a numbered option — would happily send a second action the
   *  engine has already made illegal. */
  async function liveAction(
    action: string,
    args: Record<string, unknown>,
  ): Promise<Step | null> {
    return actionGate.current.run(() => applyLiveAction(action, args));
  }
  /** {@link liveAction} without the gate — for use INSIDE an operation that
   *  already holds it (see `playBestChoice`). Never call this from an event
   *  handler. */
  async function applyLiveAction(
    action: string,
    args: Record<string, unknown>,
  ): Promise<Step | null> {
    autoHintedRef.current = false;
    setHintShown(false);
    setBlunder(null);
    // The autoplay's last search is about a position this action leaves behind.
    setLastSearch(null);
    // The cues that belong to the CHOICE rather than to its result — a shop
    // purchase, Rest vs Dig, a deck op committing. Emitted from the state the
    // action is taken FROM, before the new step lands, since that state is
    // what identifies which option this index was. See `actionCues`.
    const snd = soundRef.current;
    if (snd?.map && live) {
      const cues = actionCues(snd.map, live.state, live.decision, action, args);
      if (cues.length) snd.emit(cues);
    }
    const { step: s, error } = await api.live.action(action, args);
    setLive(s);
    setLiveError(error ?? null);
    // Illegal moves create no node; legal ones extend (or fork) the tree.
    if (!error) await refreshTimeline();
    // Autobattle mode: the pick is committed, so the champion's is now safe to
    // reveal — then skip the combats up to the next run-level decision.
    if (autobattleOn && !error && !COMBAT_VERBS.has(action)) {
      try {
        setAutobattleVerdict(await api.live.autobattleVerdict());
      } catch {
        setAutobattleVerdict(null); // never block the autobattle on a missing verdict
      }
      await advanceAutobattle();
      return error ? null : s;
    }
    // Live regret coach: a combat move's quality is judged against the MCTS
    // line for the position it was made from. The endpoint 409s for a
    // non-combat parent — caught and ignored.
    if (coachOn && !error && COMBAT_VERBS.has(action)) {
      try {
        const r = await api.live.lastMoveRegret({});
        // Show after every assessable combat move — blunder, suboptimal, or
        // best — not only flagged blunders. The error payload (no MCTS scores)
        // carries no verdict, so skip it.
        setBlunder(r.error ? null : r);
      } catch {
        // 409 = the prior decision had no counterfactual (e.g. forced move).
        setBlunder(null);
      }
    }
    return error ? null : s;
  }
  /** Take the policy's recommended option on the current choice screen — the
   *  twin of combat's "play best (G)" / "autoplay (A)". Works through card
   *  rewards, shops, events, the map, rest sites, … so you can advance toward
   *  the boss with the same key the agent would use, and through an in-combat
   *  card choice (Discover, a pile search), which the backend answers with the
   *  same combat search that scored the screen's chips. Combat card play and
   *  A/G are owned by Board; there's nothing to pick at game over.
   *
   *  An in-combat MULTI-pick (Dredge's "choose up to 3") is several engine
   *  decisions: each accepted index accumulates and re-opens the identical
   *  screen, so one press plays it out to completion. Stopping half-way would
   *  leave the engine's `picked` set out of step with the screen's own
   *  checkboxes, and the next human Confirm would submit on top of it.
   *
   *  The first-legal-option fallback is for screens the backend genuinely can't
   *  recommend on (the C# engine) — it is a blind pick, so anything worth
   *  advising should answer through auto-action instead (bug
   *  wb-31c2d79f995fe1a6: an in-combat choice used to land here and take card 0
   *  while the panel starred another card). It only runs before we've acted;
   *  once a multi-pick is under way, a screen the backend has stopped
   *  recommending on is one to hand back to the player, not to guess at. */
  async function playBestChoice() {
    if (!live || autoRunning) return;
    if (live.decision === "combat_play" || live.decision === "game_over") return;
    // The gate spans the WHOLE ask-then-dispatch, not just the dispatch. G/A are
    // plain `keydown`, so holding the key re-enters this at the OS repeat rate;
    // gating only the dispatch would still let a re-entry ask the policy about
    // the screen the first press is leaving and then send that stale answer to
    // an engine which has moved on (bug wb-12ce3fe830de0bb1). Inside, the
    // un-gated `applyLiveAction` — `liveAction` would deadlock against us.
    const release = actionGate.current.enter();
    if (!release) return;
    try {
      // Bounded well above any real max_select; a choice that somehow kept
      // re-opening must not spin the UI.
      for (let step = 0; step < 12; step++) {
        const rec = await api.live.autoAction();
        if (!rec) {
          if (step === 0) {
            const enabled = (live.state.options ?? []).filter((o) => o.enabled);
            if (enabled.length > 0) await applyLiveAction(enabled[0].action, enabled[0].args);
          }
          return;
        }
        const after = await applyLiveAction(rec.action, rec.args);
        // Resolved (or errored) — anything but a still-open in-combat multi-pick
        // is one action per press, exactly as before.
        if (!after || !isCombatChoiceStep(after) || (after.state.max_select ?? 1) <= 1) return;
      }
    } finally {
      release();
    }
  }
  async function seekTo(nodeId: number) {
    if (seekBusy) return;
    autoHintedRef.current = false;
    setHintShown(false);
    setLiveError(null);
    setBlunder(null);
    setSeekBusy(true);
    try {
      const { step } = await api.live.seek(nodeId);
      setLive(step);
      await refreshTimeline();
    } finally {
      setSeekBusy(false);
    }
  }
  /** Rewind the LIVE game to the room-entry save of the floor the user is
   *  looking at in the replay timeline. The game only saves at room entry /
   *  post-combat, so this lands at the viewed floor's start. */
  async function bridgeSeekToViewedFloor() {
    const floor = entries[idx]?.floor;
    if (floor == null || bridgeSeekBusy) return;
    setBridgeSeekBusy(true);
    setBridgeError(null);
    try {
      const { snapshots } = await api.bridge.snapshots();
      // Newest room-entry snapshot at that floor; else the nearest below it.
      const entriesAtOrBelow = (snapshots ?? [])
        .filter((sn) => !sn.post_combat && sn.floor <= floor)
        .sort((a, b) => a.seq - b.seq);
      const exact = entriesAtOrBelow.filter((sn) => sn.floor === floor).pop();
      const target = exact ?? entriesAtOrBelow.pop();
      if (!target) {
        setBridgeError(`no save point at or below floor ${floor} in this run`);
        return;
      }
      await api.bridge.seek(undefined, target.id);
    } catch (e) {
      setBridgeError(String(e));
    } finally {
      setBridgeSeekBusy(false);
    }
  }
  async function openBridgeTree() {
    setBridgeError(null);
    try {
      setBridgeTimeline(await api.bridge.tree());
      setBridgeTreeOpen(true);
    } catch (e) {
      setBridgeError(String(e));
    }
  }
  async function bridgeSeekTo(nodeId: number) {
    if (bridgeSeekBusy) return;
    // A clicked decision node has no save point of its own — walk up to the
    // nearest ancestor that does (room entry / post-combat), so every node
    // in the tree is a valid rewind gesture.
    let target = nodeId;
    if (bridgeTimeline) {
      const byId = new Map(bridgeTimeline.nodes.map((n) => [n.id, n]));
      let cur = byId.get(nodeId);
      while (cur && cur.seekable === false && cur.parent_id != null) {
        cur = byId.get(cur.parent_id);
      }
      if (!cur || cur.seekable === false) {
        setBridgeError("no save point at or before that node yet");
        return;
      }
      target = cur.id;
    }
    setBridgeSeekBusy(true);
    setBridgeError(null);
    try {
      await api.bridge.seek(target);
      // The game rewinds in-process (<2 s); re-fetch the tree so the active
      // path moves. The follow poll picks up the log fork on its own.
      setBridgeTimeline(await api.bridge.tree());
    } catch (e) {
      setBridgeError(String(e));
    } finally {
      setBridgeSeekBusy(false);
    }
  }
  async function fetchHint(): Promise<Score[]> {
    hintInFlight.current = true;
    setHintBusy(true);
    try {
      const r = await api.live.hint({ ...hintConfigRef.current });
      setLive((s) => (s ? { ...s, scores: r.scores, pv: r.pv } : s));
      return r.scores;
    } finally {
      hintInFlight.current = false;
      setHintBusy(false);
    }
  }
  async function liveHint() {
    setHintShown(true);
    await fetchHint();
  }
  // `fromBridge` is an argument, not read from state: `liveSearch` sets
  // `searchFromBridge` and calls this in the same tick, before the update
  // lands, so reading state here searched the workbench run instead of the game.
  async function openSearchTree(
    config: import("./api").MctsConfig = searchConfig,
    fromBridge: boolean = searchFromBridge,
  ) {
    setSearchConfig(config);
    setSearchOpen(true);
    setSearchBusy(true);
    try {
      const tree = fromBridge
        ? await api.bridge.searchTree(config.sims)
        : mode === "play"
          ? await api.live.searchTree({ ...config })
          : await api.searchTree(idx, { ...config });
      setSearchTree(tree);
    } catch (e) {
      setSearchTree({ nodes: [], root: null, pv: [], error: String(e) });
    } finally {
      setSearchBusy(false);
    }
  }
  async function playRecommended() {
    if (!live || live.decision !== "combat_play" || hintInFlight.current) return;
    const scores = live.scores.length > 0 ? live.scores : await fetchHint();
    const a = bestAction(scores);
    if (!a || a.kind === "unsupported") return;
    if (a.kind === "end") {
      await liveAction("end_turn", {});
      return;
    }
    if (a.kind === "potion") {
      await liveAction("use_potion", a.targetIndex != null ? { potion_index: a.slot, target_index: a.targetIndex } : { potion_index: a.slot });
      return;
    }
    if (a.targetIndex != null) setLiveAttack({ target: a.targetIndex, nonce: Date.now() });
    await liveAction(
      "play_card",
      a.targetIndex != null ? { card_index: a.cardIndex, target_index: a.targetIndex } : { card_index: a.cardIndex },
    );
  }

  async function autoplay() {
    if (autoRunning || !live || live.decision !== "combat_play") return;
    setAutoRunning(true);
    autoRunningRef.current = true;
    stopAutoRef.current = false;
    setAutoPaused(false);
    autoHintedRef.current = true; // suppress the auto-fetch effect during the loop (autoplay fetches itself)
    let played = 0; // cards/potions played during THIS press
    try {
      for (let i = 0; i < 20; i++) {
        if (stopAutoRef.current) break; // Esc interrupted
        const scores = await fetchHint();
        const a = bestAction(scores);
        if (!a || a.kind === "unsupported") break; // nothing dispatchable
        if (a.kind === "end") {
          // Best move is to end the turn. If we already played this press, stop
          // here so the user can review the spent turn (press A again to end).
          // If we played nothing (the turn is already done), end it now — so
          // repeated A presses march turn-by-turn through the whole game.
          if (played === 0) {
            const { step: s } = await api.live.action("end_turn", {});
            setLive(s);
          }
          break;
        }
        const [actName, actArgs] =
          a.kind === "potion"
            ? ["use_potion", a.targetIndex != null ? { potion_index: a.slot, target_index: a.targetIndex } : { potion_index: a.slot }] as const
            : ["play_card", a.targetIndex != null ? { card_index: a.cardIndex, target_index: a.targetIndex } : { card_index: a.cardIndex }] as const;
        if (a.kind === "play" && a.targetIndex != null) setLiveAttack({ target: a.targetIndex, nonce: Date.now() });
        const { step: s, error } = await api.live.action(actName, actArgs);
        setLive(s);
        played++;
        if (error || s.decision !== "combat_play") break; // illegal move or combat ended
      }
    } finally {
      // Clear the latch so the auto-hint effect re-fires for the resulting
      // state (otherwise the panel stays on "thinking…" until a manual action).
      autoHintedRef.current = false;
      autoRunningRef.current = false;
      setAutoRunning(false);
      await refreshTimeline();
    }
  }

  /** Drive the CURRENT run to its end through one shared loop, painting each
   *  step and refreshing the branch tree so it progresses live; Stop/Esc pauses
   *  (run stays open + resumable). Each iteration applies ONE unified stepper
   *  decision server-side (combat = the combat MCTS agent always; off-combat =
   *  the selected brain) via `api.live.runModeStep`. Bounded by a hard step
   *  cap so a non-progressing state can't spin forever. */
  async function runAutoplay(startStep?: Step) {
    // `startStep` is the just-started run's step: when launched from a fresh
    // start, the `live` state set a tick earlier is NOT yet visible in this
    // closure, so relying on it would make the loop no-op. Resume/F-key call
    // with no arg and use the already-rendered `live`.
    const start = startStep ?? live;
    if (!start || autoRunning) return;
    // Resuming a previously-stopped run keeps the elapsed clock and step count
    // ticking forward; a fresh launch resets them (and the debug log).
    const resuming = autoPaused;
    setAutoRunning(true);
    autoRunningRef.current = true;
    stopAutoRef.current = false;
    setAutoPaused(false);
    autoHintedRef.current = true; // suppress the auto-fetch effect; we drive hints
    let cur: Step = start;
    // Carry the run's floor/act forward (combat states omit them) so the
    // progress bar can estimate how far through the run we are.
    const trackProgress = (s: Step) => {
      const f = s.state?.floor;
      const a = s.state?.act;
      if (f != null) setAutoFloor(f);
      if (a != null) setAutoAct(a);
    };
    let stepCount = resuming ? autoStepNum : 0;
    if (resuming) {
      autoRunStartRef.current = Date.now() - autoRunElapsed * 1000;
    } else {
      autoRunStartRef.current = Date.now();
      setAutoRunElapsed(0);
      setAutoStepNum(0);
      setAutoFloor(null);
      setAutoAct(null);
      if (debugEnabledRef.current) setDebugLog([]);
      setLastSearch(null);
      pushDebug(settingsLine());
      pushDebug(`▶ play to end · brain=${offCombatBrain} · combat=${combatPreset}`);
    }
    trackProgress(start);
    // Throttle the branch-tree refresh: a full `GET /timeline` every single step
    // is O(N) fetches of a growing payload over a several-hundred-step run. Paint
    // the first step immediately for liveness, then at most every Nth step; the
    // `finally` always does one final full refresh (so a paused/finished tree is
    // exact). The tree still visibly progresses during the run.
    const TIMELINE_STRIDE = 6;
    let sinceRefresh = TIMELINE_STRIDE; // force a refresh on the first step
    const liveRefresh = async () => {
      if (++sinceRefresh < TIMELINE_STRIDE) return;
      sinceRefresh = 0;
      await refreshTimeline();
    };
    try {
      for (let i = 0; i < 50000; i++) {
        // Mark this step's start so the progress bar's elapsed readout resets
        // and the timer interval ticks off the current (possibly slow) step.
        autoStepStartRef.current = Date.now();
        if (stopAutoRef.current) {
          if (cur.decision !== "game_over") setAutoPaused(true);
          break;
        }
        if (cur.decision === "game_over") break;
        stepCount += 1;
        setAutoStepNum(stepCount);
        setAutoStepElapsed(0);
        const r = await api.live.runModeStep();
        cur = r.step;
        setLive(r.step);
        // Off-combat steps carry no search; keep the last combat one on screen
        // rather than blanking the panel between fights.
        if (r.search) setLastSearch(r.search);
        trackProgress(r.step);
        if (r.debug?.summary) pushDebug(debugStepLine(r.debug));
        await liveRefresh();
        if (r.error) { setLiveError(`run stopped: ${r.error}`); break; }
        if (r.done || r.over) break;
      }
    } catch (e) {
      // An unexpected throw (e.g. a network blip mid-step) leaves the run alive
      // and resumable. Mark it paused so the in-mode Resume affordance appears —
      // otherwise the only resume path (F / Board) would strand the run in a
      // half-stepped state.
      setLiveError(`autoplay error: ${e}`);
      if (cur.decision !== "game_over") setAutoPaused(true);
    } finally {
      autoHintedRef.current = false;
      autoRunningRef.current = false;
      setAutoRunning(false);
      pushDebug(
        cur.decision === "game_over"
          ? `■ run over after ${stepCount} steps (${fmtDuration((Date.now() - autoRunStartRef.current) / 1000)})`
          : `⏸ paused after ${stepCount} steps`,
      );
      await refreshTimeline();
    }
  }
  /** In-combat "play to end (F)" button + F key: drive the CURRENT run to its
   *  end via the unified stepper (`runAutoplay` loops `runModeStep`). */
  async function playToEnd() {
    if (autoRunning || !live || live.decision === "game_over") return;
    await runAutoplay();
  }
  /** "Watch the AI play" launcher on the start screen. Posts the current two-axis
   *  run-mode config (off-combat brain x combat sim budget), then starts a
   *  fresh parity run from the seed for EVERY brain (heuristic and PPO both
   *  drive through the same `RunStepper` + unified stepper) — unless a
   *  hand-played run is already open, in which case that run is kept as-is
   *  and just looped to completion (no restart). Resume (paused mid-run)
   *  re-enters `runAutoplay` directly, bypassing this launcher entirely. */
  async function launchAutoplay() {
    if (autoRunning) return;
    // Claim the running flag SYNCHRONOUSLY (before any await) so a second click
    // during the start round-trip — while `live` is still null and the launcher
    // is still mounted — can't fire a duplicate startFaithful/start.
    setAutoRunning(true);
    autoRunningRef.current = true;
    autoHintedRef.current = false;
    setLiveError(null);
    setEndedRun(null);
    let started: Step | undefined;
    try {
      if (!live) {
        // Start the run FIRST — `startFaithful` starts the run with the
        // server's default run-mode config, so it must create the session
        // before the `runMode` call below applies the user's actual brain
        // selection. Posting `runMode` first also 400s on a fresh run:
        // there is no live session yet for it to configure.
        const runSeed = seed || generateSeed();
        if (runSeed !== seed) setSeed(runSeed);
        started = await api.live.startFaithful({
          character,
          seed: runSeed,
          ascension,
          sims: monsterSims,
          boss_sims: bossSims,
          threads: faithfulThreads,
        });
        setMode("play");
        resetReplayView();
        setLive(started);
        setRegret(null);
        regretLiveRef.current = null;
        await refreshTimeline();
      } else {
        // A hand-played run is already open — keep it, just switch to the
        // play screen and let runAutoplay loop runModeStep on it below.
        setMode("play");
      }
      // Apply the selected run-mode config (off-combat brain + combat sim
      // budget) now that a live session is guaranteed to exist — this is
      // the single place that posts it (the brain `<select>` only updates
      // local state, see below).
      await api.live.runMode({
        off_combat_brain: offCombatBrain,
        combat_sims: monsterSims,
        combat_boss_sims: bossSims,
      });
      setAppliedBudget({ monster: monsterSims, boss: bossSims });
    } catch (e) {
      setLiveError(`play-to-end failed: ${e}`);
      setAutoRunning(false);
      autoRunningRef.current = false;
      return;
    }
    // Keep the flag claimed straight through into runAutoplay (its `finally`
    // releases it). `started` carries the fresh step — `live` isn't yet visible
    // in this closure. runAutoplay re-reads the closure's stale `autoRunning`
    // (false at this render), so its guard still lets it proceed.
    await runAutoplay(started);
  }

  /** Re-post the combat sim budget to a run that is already open — the
   *  "this run is going badly, search harder" path. Takes effect from the next
   *  decision: the server pushes it straight onto the live agent, which keeps
   *  its RNG and decision counter, so a paused run resumes where it stopped
   *  rather than replaying seeds it already spent.
   *
   *  Safe to press while the loop is running; the in-flight step finishes at
   *  the old budget. */
  async function applyBudget() {
    if (!live) return;
    try {
      await api.live.runMode({ combat_sims: monsterSims, combat_boss_sims: bossSims });
      setAppliedBudget({ monster: monsterSims, boss: bossSims });
      pushDebug(`⚙ combat budget → mon ${monsterSims} / boss ${bossSims}`);
    } catch (e) {
      setLiveError(`could not change the sim budget: ${e}`);
    }
  }

  useShortcuts({
    mode, live, step, count, idx,
    onScrub: (d) =>
      setIdx((i) =>
        d === "home" ? 0 : d === "end" ? Math.max(0, count - 1) : Math.max(0, Math.min(count - 1, i + d)),
      ),
    onRecommend: recommend,
    onChoose: liveAction,
    // G / A ask the run policy for the answer to the screen you are on. Both
    // are press-first, so neither follows the hints switch — same as the
    // buttons they mirror.
    onPlayBest: playBestChoice,
    onPlayToEnd: playToEnd,
    autoRunning,
    onStart: startLive,
    onToggleHelp: () => setHelpOpen((o) => !o),
  });

  const here = entries[idx];

  // The top bar's one-shot verbs, as data (topbar/commands.ts decides what
  // is offered and why an item is disabled). Rebuilt per render: it is a
  // handful of objects, and the handlers close over the current state.
  const commands = buildCommands({
    mode,
    hasLive: !!live,
    liveDecision: live?.decision ?? null,
    stepDecision: step?.decision ?? null,
    replayCount: count,
    advisors,
    stateImport,
    staticMode: isStaticMode(),
    analyzeInTopbar,
    resume,
    bridgeConnected,
    bridgeDirect: !!bridgeUrl,
    viewedFloor: here?.floor ?? null,
    busy: { simlabFromHere: simlabBusy, recommend: recommendBusy, playFromHere: playFromHereBusy, bridgeSeek: bridgeSeekBusy },
    actions: {
      save: () => { void saveLive(); },
      load: () => { void openLoadDialog(); },
      importState: () => { setImportFromLive(false); setImportOpen(true); },
      importLive: () => { setImportFromLive(true); setImportOpen(true); },
      endRun: () => { void endRun(); },
      analyze: () => { void openAnalysis(); },
      regretReview: () => openRegret(mode === "play"),
      runReview: () => openWpa(mode === "play"),
      searchTree: () => { void openSearchTree(); },
      simlabFromHere: () => { void simlabFromHere(); },
      recommend: () => { void recommend(); },
      playFromHere: () => { void playFromHere(); },
      bridgeTree: () => { void openBridgeTree(); },
      liveSearch: () => { setSearchFromBridge(true); void openSearchTree(searchConfig, true); },
      bridgeSeek: () => { void bridgeSeekToViewedFloor(); },
    },
  });
  // "Where am I": act + floor. The act NAME is the only thing that says
  // which Act 1 this run walks — STS2 ships two models at index 0
  // (Overgrowth / Underdocks, GH #544) and the two look nothing alike, so
  // "Act 1" alone leaves a bug report ambiguous.
  const whereLabel = (() => {
    const src = mode === "replay" ? here : mode === "play" && live ? live.meta : null;
    if (!src) return null;
    const act = src.act_name ?? (src.act != null ? `Act ${src.act}` : "");
    return `${act}${src.floor != null ? ` · floor ${src.floor}` : ""}` || null;
  })();

  const reportId = new URLSearchParams(window.location.search).get("report");
  if (reportId) return <BugReportView id={reportId} />;

  return (
    <EffectsProvider autoplaying={autoRunning}>
    <SoundProvider>
    <SoundBridge soundRef={soundRef} live={live} mode={mode} />
    <div className="wb-shell"
      style={{ gridTemplateColumns: sideVisible && !narrow ? "1fr 320px" : "1fr" }}>
      <div className="wb-main">
        {phoneLandscape && (
          <button type="button" className="wb-chrome-toggle" onClick={() => setChromeOpen((v) => !v)}
            aria-label="workbench tools" aria-expanded={chromeOpen}
            title={chromeOpen ? "hide the workbench tool row" : "show the workbench tool row"}>
            {chromeOpen ? "✕" : "≡"}
          </button>
        )}
        <div className={`mono wb-topbar${phoneLandscape ? " wb-topbar--sheet" : ""}`} hidden={topbarHidden}>
        {/* View switch · AI chips · context · Run/Analyze/Tools menus · where.
            What each menu offers is `buildCommands` (topbar/commands.ts); the
            bar itself only lays it out. */}
        <TopBar
          mode={mode} setMode={setMode}
          showViewToggle={shouldShowViewToggle({ hasLive: !!live, hasEndedRun: !!endedRun, replayCount: count })}
          replayCount={count}
          commands={commands}
          chips={{
            hints: mode === "play" && advisors ? {
              on: hints.on, toggle: () => hints.setOn(!hints.on),
              title: "AI hints: show the machine's opinion WITHOUT being asked — the PPO policy panel, auto combat hints, the EV analyzers, the synergy/archetype chips and the map's planned route. Off = play unadvised; the press-first tools (hint, coach, the reviews, search) still work.",
            } : undefined,
            autobattle: mode === "play" && live && advisors && runMode ? {
              on: autobattleOn,
              toggle: () => {
                // Leaving autobattle mode drops the drill state so re-entering
                // starts a fresh diff window rather than resuming a stale one.
                setAutobattleOn((v) => !v);
                setAutobattleDiff(null);
                setAutobattleVerdict(null);
                setAutobattleOver({ over: false, won: false });
              },
              title: "autobattle mode: the agent plays the fights (watch them play out, or skip them) and you make only the run-level decisions — with a change-diff at each stop and the champion's pick revealed after you commit",
            } : undefined,
            coach: mode === "play" && live && advisors ? {
              on: coachOn, toggle: () => { setCoachOn((v) => !v); setBlunder(null); },
              title: "regret coach: warn right after a combat move that cost win-probability vs the MCTS line",
            } : undefined,
          }}
          replay={mode === "replay" ? {
            runs, loadRun: (path) => { void loadRun(path); }, loadedPath,
            follow, setFollow, followIsLive, followFromUrl: followFromUrlRef.current, bridgeError,
          } : undefined}
          tools={{
            simlabLink: advisors && simlab,
            debug: { on: debugEnabled, toggle: toggleDebug },
            onHelp: () => setHelpOpen((o) => !o),
            ...(isStaticMode()
              ? { onBridge: () => setBridgeOpen(true),
                  bridgeState: !bridgeUrl ? "off" as const : bridgeConnected ? "connected" as const : "unreachable" as const }
              : {}),
            controls: [
              bugReports ? (
                <BugReportButton key="bug" mode={mode} step={idx}
                  screen={(live ?? step)?.decision ?? null} runPath={loadedPath}
                  onCaptured={(r) => { setBugError(null); setBugResult(r); }}
                  onFailed={(m) => { setBugResult(null); setBugError(m); }} />
              ) : null,
              <BenchButton key="bench" inCombat={mode === "play" && live?.decision === "combat_play"} />,
              isStaticMode() ? <OfflineButton key="offline" /> : null,
              <EffectsToggle key="fx" />,
              <SoundToggle key="sound" />,
            ],
          }}
          where={whereLabel}
          whereTitle={mode === "play" ? "the act this run is walking · Act 1 has two models (Overgrowth / Underdocks)" : undefined}
          palette={{ open: paletteOpen, setOpen: setPaletteOpen }}
          trailing={
            /* Outside the scroller, so the way back from a drawer that is covering
               the board is always on screen. On a tablet in landscape the 320px
               panel costs the game a third of its width; this hides it. */
            <button className="wb-side-toggle" onClick={() => setSideOpen(!sideVisible)}
              aria-label="side panel" aria-pressed={sideVisible}
              title={sideVisible ? "hide the side panel (map, history, reasoning)"
                : "show the side panel (map, history, reasoning)"}>
              {sideVisible ? "▨" : "▤"}
            </button>
          }
        />
        </div>
        {/* The screen column: the active screen (scrolling) with the PPO dock
            parked under it. The regret / WPA review docks are `position: fixed`
            over the window's bottom edge, so the WHOLE column reserves their
            height — not just the scroll box inside it, which used to leave the
            PPO dock under them. */}
        <div style={{ position: "relative", flex: 1, minHeight: 0, display: "flex", flexDirection: "column",
          paddingBottom: regretHeight + wpaHeight }}>
          {/* Replay's copy of the error banner. The other two live inside the
              play-mode branches, so a failed "play from here" in replay mode set
              liveError and NOTHING rendered it — the button read as inert. */}
          {mode === "replay" && liveError && (
            <div className="mono" style={{ margin: "8px 12px 0", padding: "4px 8px", borderRadius: 4,
              color: "var(--hp)", border: "1px solid var(--hp)", fontSize: 11 }}>
              ⚠ {liveError}
            </div>
          )}

          {/* What the imported position does NOT reproduce. Sits on the board
              rather than in the dialog that is now closed: a fidelity caveat is
              only useful while you are looking at the thing it qualifies. */}
          {restoreProgress && (
            <div className="mono" role="status" style={{ margin: "8px 12px 0", padding: "6px 8px", borderRadius: 4,
              color: "var(--muted)", border: "1px solid var(--border)", fontSize: 11 }}>
              Restoring run {restoreProgress.done}/{restoreProgress.total}…
            </div>
          )}
          {mode === "play" && importNotes.length > 0 && (
            <div className="mono" style={{ margin: "8px 12px 0", padding: "6px 8px", borderRadius: 4,
              color: "var(--warn, #c90)", border: "1px solid var(--warn, #c90)", fontSize: 11,
              display: "flex", gap: 8, alignItems: "flex-start" }}>
              <div style={{ flex: 1 }}>
                <b>imported position — not everything carried over:</b>
                <ul style={{ margin: "4px 0 0", paddingLeft: 16 }}>
                  {importNotes.map((n, i) => <li key={i}>{n}</li>)}
                </ul>
              </div>
              <button onClick={() => setImportNotes([])} aria-label="Dismiss import notes"
                style={{ border: "none", background: "none", color: "inherit", cursor: "pointer" }}>
                ✕
              </button>
            </div>
          )}

          {/* Live "play to end" progress bar. A run-mode step blocks for seconds
              (boss-sim card rewards, combat MCTS) with no streamable progress, so
              the % / ETA are estimated from how far through the run we are (the
              current floor vs a typical run length) rather than a true fraction of
              work done. `live` still holds the screen being computed-from, so its
              decision labels the work. Rendered as an absolute overlay (not a flow
              sibling) so it appearing/disappearing doesn't shift the board below. */}
          {mode === "play" && autoRunning && (() => {
            const d = live?.decision;
            const label =
              d === "combat_play" ? "searching combat (MCTS)…"
              : d === "card_reward" ? "evaluating card rewards…"
              : d === "card_select" ? "choosing card…"
              : d === "map_select" ? "choosing path…"
              : d === "shop" ? "shopping…"
              : d === "event_choice" ? "resolving event…"
              : "thinking…";
            // Estimate run progress from the current floor; clamp shy of 100% so
            // a longer-than-typical run doesn't read as finished early.
            const frac = autoFloor != null
              ? Math.min(Math.max(autoFloor / EST_TOTAL_FLOORS, 0.02), 0.99)
              : 0.02;
            const pct = Math.round(frac * 100);
            const eta = autoFloor != null && autoRunElapsed > 0
              ? (autoRunElapsed * (1 - frac)) / frac
              : null;
            const estSteps = autoFloor != null && autoStepNum > 0
              ? Math.round(autoStepNum / frac)
              : null;
            const where = [autoAct != null ? `A${autoAct}` : null, autoFloor != null ? `floor ${autoFloor}` : null]
              .filter(Boolean).join(" ");
            return (
              <div className="mono" style={{ position: "absolute", top: 0, left: 0, right: 0, zIndex: 5,
                display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
                padding: "5px 12px", borderBottom: "1px solid var(--border)", fontSize: 12,
                background: "var(--bg)", boxShadow: "0 2px 6px rgba(0,0,0,0.35)" }}>
                <span style={{ color: "var(--muted)" }}>
                  ▶ {label} · step {autoStepNum}{estSteps ? ` / ~${estSteps}` : ""}
                  {where ? ` · ${where}` : ""} · ~{pct}% · {fmtDuration(autoRunElapsed)} elapsed
                  {eta != null ? ` · ETA ${fmtDuration(eta)}` : ""}
                  {autoStepElapsed >= 1 ? ` · this step ${autoStepElapsed.toFixed(1)}s` : ""}
                </span>
                <div className="auto-progress" title={`~${pct}% (estimated from floor ${autoFloor ?? "?"} of ~${EST_TOTAL_FLOORS})`}
                  style={{ flex: 1, minWidth: 80, height: 6, borderRadius: 3,
                    overflow: "hidden", background: "var(--field-bg)" }}>
                  <div style={{ height: "100%", width: `${pct}%`, borderRadius: "inherit",
                    background: "var(--accent)", transition: "width 0.3s ease" }} />
                </div>
                <button type="button" onClick={() => { stopAutoRef.current = true; }}
                  title="stop the run (Esc) — it stays open and you can resume"
                  style={{ padding: "1px 8px", border: "1px solid var(--hp)", borderRadius: 4,
                    background: "var(--bg)", color: "var(--hp)", cursor: "pointer", font: "inherit" }}>
                  ⏹ stop
                </button>
              </div>
            );
          })()}
          {/* Combat budget of a LIVE play-to-end run, editable in place.
              `appliedBudget` is what marks the run as run-mode driven, so this
              never appears over a hand-played run that has no combat agent to
              retarget. Deliberately outside the tool band below, which hides
              itself in combat — combat is exactly where you notice the run
              going badly and want to search harder. */}
          {mode === "play" && live && live.decision !== "game_over" && appliedBudget && (
            <div className="mono" style={{ display: "flex", gap: 8, alignItems: "center",
              padding: "4px 16px", borderBottom: "1px solid var(--border)", flexWrap: "wrap", fontSize: 12 }}>
              <CombatBudget
                preset={combatPreset} monsterSims={monsterSims} bossSims={bossSims}
                threads={faithfulThreads}
                onPreset={setCombatPreset} onMonsterSims={setMonsterSims} onBossSims={setBossSims}
                pending={
                  monsterSims !== appliedBudget.monster || bossSims !== appliedBudget.boss
                    ? { monster: monsterSims, boss: bossSims }
                    : null
                }
                onApply={applyBudget}
              />
              <span style={{ color: "var(--muted)", fontSize: 11 }}>
                searching at {appliedBudget.monster}/{appliedBudget.boss}
              </span>
            </div>
          )}
          {(() => {
            // Persistent game HUD band above every screen (HP / gold / relics
            // persist across the run) — the drawn in-viewport bar (GameHud),
            // preceded by a slim WORKBENCH tool row for the controls that do
            // not exist in the real game's band (autoplay, analyze). Map and
            // deck stay in the band: the game's bar has both.
            const cur = (mode === "play" ? live : step)?.state;
            const p = cur?.player;
            if (!p || p.hp == null) return null;
            const deck = p.deck ?? [];
            // Off-combat autoplay: take the policy's best option here, or play
            // the whole run out. Combat has its own controls (Board); these
            // only show on non-combat play screens.
            const showAuto = mode === "play" && live && runMode
              && cur.decision !== "combat_play" && cur.decision !== "game_over";
            return (
              <>
                {(showAuto || (deck.length > 0 && !analyzeInTopbar)) && (
                  <div className="mono wb-runbar" style={{ display: "flex", gap: 8, alignItems: "center",
                    padding: "4px 16px", borderBottom: "1px solid var(--border)", flexWrap: "wrap", fontSize: 12 }}>
                    {showAuto && (
                      // Reserve width for the widest control set (best ·
                      // play-to-end · stop) so the analyze control after this
                      // group doesn't jump sideways when stop / resume swap.
                      <span style={{ display: "inline-flex", gap: 4, minWidth: 300 }}>
                        <button type="button" onClick={playBestChoice} disabled={autoRunning}
                          title="take the recommended best option for this screen (G / A)"
                          style={{ padding: "2px 8px", border: "1px solid var(--border)", borderRadius: 4,
                            background: "var(--bg)", cursor: autoRunning ? "default" : "pointer", font: "inherit" }}>
                          ▶ best (G)
                        </button>
                        <button type="button" onClick={playToEnd} disabled={autoRunning}
                          title="play the whole run out — map, rewards, fights, bosses — until it ends or you press Esc (F)"
                          style={{ padding: "2px 8px", border: "1px solid var(--border)", borderRadius: 4,
                            background: "var(--bg)", cursor: autoRunning ? "default" : "pointer", font: "inherit" }}>
                          ⏭ play to end (F)
                        </button>
                        {autoRunning && (
                          <button type="button" onClick={() => { stopAutoRef.current = true; }}
                            title="stop autoplay (Esc)"
                            style={{ padding: "2px 8px", border: "1px solid var(--hp)", borderRadius: 4,
                              background: "var(--bg)", color: "var(--hp)", cursor: "pointer", font: "inherit" }}>
                            ⏹ stop (Esc)
                          </button>
                        )}
                        {autoPaused && !autoRunning && live && live.decision !== "game_over" && (
                          <button type="button" onClick={() => runAutoplay()}
                            title="resume playing this run to the end"
                            style={{ padding: "2px 8px", border: "1px solid var(--border)", borderRadius: 4,
                              background: "var(--bg)", cursor: "pointer", font: "inherit" }}>
                            ⏵ resume
                          </button>
                        )}
                      </span>
                    )}
                    {deck.length > 0 && (
                      <button type="button" onClick={openAnalysis} title="grade this deck: damage / block / velocity"
                        style={{ padding: "2px 8px", border: "1px solid var(--border)", borderRadius: 4,
                          background: "var(--bg)", cursor: "pointer", font: "inherit" }}>
                        📊 analyze
                      </button>
                    )}
                  </div>
                )}
                {/* Map is always available from the band — the act map can be
                    inspected from any screen (combat, card reward, shop, …). */}
                <GameHud state={cur} onOpenMap={openMap}
                  onOpenDeck={deck.length > 0 ? () => { emitUi("screenOpen"); setDeckOpen(true); } : undefined} />
              </>
            );
          })()}
          {/* Inner scroll region: the persistent status strip above stays pinned
              while only the active screen scrolls. Combat's Board fills this box
              (height:100%) so its End Turn action bar pins to the bottom. */}
          {/* `position: relative` anchors the room-reveal overlay below to this
              box, so a chest covers the screen you landed on and nothing else. */}
          <div style={{ position: "relative", flex: 1, minHeight: 0, overflow: "auto" }}>
          {(() => {
            // A treasure room resolves inside the engine, so its reveal rides
            // on the NEXT decision's state — live in play, the viewed step in
            // replay. Keyed so the next chest opens the panel again and a mere
            // re-render does not.
            const src = mode === "play" ? live : step;
            const key = revealKeyOf(src?.state, mode === "play" ? "live" : `step${step?.idx ?? 0}`);
            if (!key || key === dismissedReveal || !src?.state.room_reveal) return null;
            return (
              <TreasureReveal
                reveal={src.state.room_reveal}
                onDismiss={() => setDismissedReveal(key)}
              />
            );
          })()}
          {mode === "mcts" ? (
            <MctsRuns onOpen={openMctsRun} />
          ) : mode === "play" ? (
            <>
              {!live && endedRun && (
                <RunEnded step={endedRun.step} logPath={endedRun.logPath}
                  onNewRun={startNewRunFromSummary} onReviewReplay={reviewEndedReplay} />
              )}
              {!live && !endedRun && (
                <RunLauncher
                  character={character} onCharacter={setCharacter}
                  seed={seed} onSeed={setSeed}
                  ascension={ascension} onAscension={setAscension}
                  act1={act1} onAct1={setAct1}
                  backend={backend} onBackend={setBackend} backends={backends}
                  onPlay={startLive}
                  onWatch={runMode ? launchAutoplay : undefined}
                  busy={autoRunning}
                  error={liveError}
                  offCombatBrain={offCombatBrain} onOffCombatBrain={setOffCombatBrain}
                  combat={{
                    preset: combatPreset, monsterSims, bossSims, threads: faithfulThreads,
                    onPreset: setCombatPreset, onMonsterSims: setMonsterSims, onBossSims: setBossSims,
                  }}
                />
              )}
              {live && live.decision === "combat_play" && (
                <Board step={live}
                  actionNode={liveNode}
                  onPlayCard={(ci, ti) => {
                    if (ti != null) setLiveAttack({ target: ti, nonce: Date.now() });
                    const sound = soundRef.current;
                    if (sound?.map) {
                      const cues = cardPlayCues(
                        sound.map, live.state.hand, ci, live.state.player?.name,
                      );
                      if (cues.length) sound.emit(cues);
                    }
                    liveAction("play_card", ti != null ? { card_index: ci, target_index: ti } : { card_index: ci });
                  }}
                  onEndTurn={() => liveAction("end_turn", {})}
                  onUsePotion={(slot, ti) =>
                    liveAction("use_potion", ti != null ? { potion_index: slot, target_index: ti } : { potion_index: slot })
                  }
                  onDiscardPotion={(slot) => liveAction("discard_potion", { potion_index: slot })}
                  onHint={advisors ? liveHint : undefined}
                  attack={liveAttack ?? undefined}
                  scores={live.scores}
                  onPlayRecommended={advisors ? playRecommended : undefined}
                  onAutoplay={advisors ? autoplay : undefined}
                  onAutoplayToEnd={advisors ? playToEnd : undefined}
                  onSearch={advisors ? () => openSearchTree() : undefined}
                  autoHints={autoHints}
                  onToggleHints={!advisors ? undefined : () => {
                    // The toggle only governs the *auto-fetch* effect; last
                    // fetched scores (from H / autoplay / a prior auto-hint)
                    // live on `live.scores`. Turning OFF latches against a
                    // re-fetch and clears those scores so the off state is
                    // visible; turning ON releases the latch so the auto-fetch
                    // effect fires again (otherwise re-enabling is a no-op and
                    // the toggle looks stuck off). See toggleHintsLatch.
                    const { latch, clearScores } = toggleHintsLatch(autoHints);
                    autoHintedRef.current = latch;
                    if (clearScores) setLive((s) => (s ? { ...s, scores: [], pv: null } : s));
                    hints.setOn(!autoHints);
                  }}
                  hintBusy={hintBusy || autoRunning}
                  showScores={showScores}
                  hintConfig={advisors ? hintConfig : undefined}
                  onHintConfig={setHintConfig} />
              )}
              {live && live.decision === "map_select" && (() => {
                const ms = live.state as unknown as {
                  full_map?: FullMap;
                  act?: number | null;
                  act_name?: string | null;
                  floor?: number | null;
                  choices?: { col: number; row: number }[];
                };
                // With the full graph, the map itself is the picker: highlight
                // the reachable next nodes and let the player click one. Fall
                // back to the button list only when no graph was captured.
                if (ms.full_map) {
                  const selectable = new Set((ms.choices ?? []).map((c) => `${c.col},${c.row}`));
                  return (
                    <MapView
                      rows={[]}
                      graphs={[{
                        idx: 0,
                        act: ms.act ?? null,
                        act_name: ms.act_name ?? null,
                        floor: ms.floor ?? null,
                        full_map: ms.full_map,
                      }]}
                      currentIdx={0}
                      selectable={selectable}
                      onSelect={(col, row) => { emitUi("mapSelect"); liveAction("select_map_node", { col, row }); }}
                      live
                    />
                  );
                }
                return <MapChoices step={live} onChoose={(action, args) => { emitUi("mapSelect"); return liveAction(action, args); }} />;
              })()}
              {live && live.decision !== "combat_play" && live.decision !== "map_select" && (
                <DecisionScreen
                  step={live}
                  onChoose={liveAction}
                  onNewRun={() => {
                    setLive(null); setLiveTimeline(null); setLiveError(null);
                    api.live.quit().then(refreshRunList, () => {});
                  }}
                  onReviewReplay={reviewFinishedRun}
                />
              )}
            </>
          ) : (
            <>
              {!step && <div className="mono" style={{ padding: 20, color: "var(--muted)" }}>no run loaded</div>}
              {step && step.decision === "combat_play" && (
                <Board step={step} />
              )}
              {step && step.decision === "map_select" && <MapView rows={mapRows} graphs={mapGraphs} currentIdx={idx} onJump={(i) => setIdx(i)} />}
              {step && !["combat_play", "map_select"].includes(step.decision) && (
                <DecisionScreen step={step} />
              )}
            </>
          )}
          </div>
          {/* PPO dock — a SIBLING of the scroll box, never a child of it. It
              used to ride inside it as `position: sticky`, which lays it OVER
              the bottom of whatever screen is up: on a card reward that is
              exactly where the advisor rail (the EV chips, the analyze button's
              own output) lives, so the panel simply covered the evidence it was
              sitting next to. Out here the scroll box is `flex: 1` against it
              and the screen re-fits into what is left — the dock takes space
              instead of stealing it, and it is still on screen on every
              non-combat screen without needing to be sticky. */}
          {mode === "play" && live && advisors && <PpoPolicyPanel live={live} />}
        </div>
        {count > 0 && <Timeline idx={idx} count={count} entries={entries} onSeek={(i) => setIdx(Math.max(0, Math.min(count - 1, i)))} />}
        {/* Debug drawer, docked at the bottom of the left column. A fixed-height
            flex child: the content area above (flex:1) absorbs its space, so the
            relic strip never moves. Collapsed = header only; both states are a
            fixed height so toggling never shifts anything. Available in any mode
            (incl. after a run) so the trace can be reviewed. */}
        {debugEnabled && (
          <div className="mono" style={{ flex: "none", borderTop: "1px solid var(--border)",
            fontSize: 11, background: "var(--field-bg)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "3px 12px",
              color: "var(--muted)" }}>
              <span>🐞 debug log · {debugLog.length} lines</span>
              <button type="button" onClick={() => setDebugLog([])}
                style={{ padding: "0 6px", border: "1px solid var(--border)", borderRadius: 4,
                  background: "var(--bg)", color: "var(--muted)", cursor: "pointer", font: "inherit" }}>
                clear
              </button>
              <button type="button" onClick={() => setDebugCollapsed((c) => !c)}
                title={debugCollapsed ? "expand debug log" : "collapse debug log"}
                style={{ marginLeft: "auto", padding: "0 6px", border: "1px solid var(--border)",
                  borderRadius: 4, background: "var(--bg)", color: "var(--muted)", cursor: "pointer",
                  font: "inherit" }}>
                {debugCollapsed ? "▲" : "▼"}
              </button>
            </div>
            {!debugCollapsed && (
              <div style={{ height: 140, overflow: "auto", padding: "0 12px 6px" }}>
                {debugLog.length === 0 ? (
                  <div style={{ color: "var(--muted)" }}>… no entries yet</div>
                ) : debugLog.map((line, i) => (
                  <div key={i} style={{ whiteSpace: "pre-wrap", lineHeight: 1.5,
                    color: line.includes("·") ? "var(--text)" : "var(--muted)" }}>{line}</div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      <div className={`wb-side${narrow ? " wb-side--overlay" : ""}`} hidden={!sideVisible}>
        {/* Persistent minimap: always on screen (vs the on-demand overlay) so
            the run's position and the path ahead stay visible on every screen.
            Hidden in the MCTS dashboard, which has no single active run. */}
        {mode !== "mcts" && (
          <MiniMap
            graphs={mode === "play" ? liveMapGraphs : mapGraphs}
            currentIdx={mode === "play" ? 0 : idx}
            collapsed={minimapCollapsed}
            onToggleCollapsed={() => setMinimapCollapsed((v) => !v)}
            onExpand={openMap}
            live={mode === "play"}
          />
        )}
        {mode === "mcts" ? (
          <div className="mono" style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: 16, color: "var(--muted)" }}>
            click a run to open it in the replay timeline
          </div>
        ) : mode === "play" ? (
          <>
            {/* Reserved hint region: a separate, non-scrolling area above the
                log (it does NOT overlay it). Rendered for the whole combat so
                it doesn't mount/unmount as scores come and go during autoplay
                (which caused a show/hide flicker). Scrolls internally if tall,
                folds to its header, and its bottom edge drags — see
                ReasoningDock. */}
            {live && live.decision === "combat_play" && showScores && (() => {
              // Autoplay used to blank this panel to a flat "autoplaying…"
              // to stop it flipping on every step. That also meant the MCTS
              // bars — the reason to watch a run play itself — were never
              // drawn. Now the last search is rendered instead, headed by
              // the move it produced, since it describes the position
              // BEFORE the board on screen. See searchPanel.ts.
              const view = reasoningView({
                autoRunning, scores: live.scores, pv: live.pv, lastSearch, hintBusy, autoHints,
              });
              return (
                <ReasoningDock
                  title={view.kind === "scores" ? view.title : null}
                  collapsed={reasoningCollapsed}
                  onToggle={() => setReasoningCollapsed(!reasoningCollapsed)}
                  height={reasoningHeight}
                  onResize={setReasoningHeight}
                >
                  {view.kind === "message" ? (
                    <div className="mono" style={{ color: "var(--muted)", fontSize: 11 }}>
                      {view.text}
                    </div>
                  ) : (
                    <Reasoning scores={view.scores} pv={view.pv} heading={false}
                      enemies={view.enemies ?? live.state.enemies} />
                  )}
                </ReasoningDock>
              );
            })()}
            {/* Log region: a PINNED header (errors, coach banner, branch-tree
                button, the "history" caption) over an independently-scrolling
                history list — so the warning + caption stay put as the log
                grows and only the list itself scrolls. */}
            <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
              <div style={{ flexShrink: 0, padding: "16px 16px 0" }}>
                {liveError && (
                  <div className="mono" style={{ marginBottom: 8, padding: "4px 8px", borderRadius: 4,
                    color: "var(--hp)", border: "1px solid var(--hp)", fontSize: 11 }}>
                    ⚠ {liveError}
                  </div>
                )}
                {autobattleOn && mode === "play" && live && (
                  <AutobattlePanel
                    diff={autobattleDiff}
                    verdict={autobattleVerdict}
                    busy={autobattleBusy}
                    over={autobattleOver.over}
                    won={autobattleOver.won}
                    watch={autobattleWatch}
                    onWatchChange={setAutobattleWatch}
                    onAdvance={advanceAutobattle}
                    onStop={() => { stopAutoRef.current = true; }}
                  />
                )}
                {blunder && (() => {
                  // Four bands: best move, a tie ON the line (drop within the
                  // near-tie/noise floor), a sub-threshold slip (amber ▽), or a
                  // flagged blunder (red ⚠). Near-ties are NOT "suboptimal".
                  const onLine = blunder.matched_best || blunder.drop_pct < NEAR_TIE_DROP_PCT;
                  const tone = onLine ? "var(--accent-strong)"
                    : blunder.is_blunder ? "var(--hp)" : "var(--warn, #c79a3a)";
                  return (
                    <div className="mono" style={{ marginBottom: 8, padding: "6px 10px", borderRadius: 4,
                      color: tone, border: `1px solid ${tone}`, background: "var(--field-bg)",
                      fontSize: 11, display: "flex", gap: 8, alignItems: "baseline" }}>
                      <span style={{ flex: 1 }}>
                        {onLine ? (
                          <>✓ {blunder.matched_best ? "best move" : "on the MCTS line"} — <b>{blunder.played_label}</b> ({Math.round(blunder.played_win_prob * 100)}%)</>
                        ) : (
                          <>
                            {blunder.is_blunder ? "⚠ blunder" : "▽ suboptimal"} — <b>{blunder.played_label}</b> ({Math.round(blunder.played_win_prob * 100)}%)
                            {" "}instead of <b>{blunder.best_label}</b> ({Math.round(blunder.best_win_prob * 100)}%),
                            {" "}−{blunder.drop_pct} pts
                          </>
                        )}
                      </span>
                      <button type="button" onClick={() => setBlunder(null)} aria-label="Dismiss"
                        style={{ font: "inherit", background: "none", border: "none", color: "var(--muted)",
                          cursor: "pointer" }}>✕</button>
                    </div>
                  );
                })()}
                {liveTimeline && liveTimeline.nodes.length > 1 && (
                  <button type="button" onClick={() => setTreeOpen(true)}
                    title="open the branch tree" style={{ marginBottom: 12, padding: "3px 10px",
                      border: "1px solid var(--border)", borderRadius: 4, background: "var(--bg)",
                      cursor: "pointer", font: "inherit", fontSize: 11 }}>
                    ⛶ branch tree
                    {liveTimeline.nodes.some((n) => n.child_ids.length > 1) ? " (forked)" : ""}
                  </button>
                )}
                {liveTimeline && liveTimeline.nodes.length > 0 && (
                  <div className="mono" style={{ display: "flex", alignItems: "baseline",
                    justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 0.5 }}>
                      history
                    </span>
                    <span style={{ fontSize: 10, color: "var(--muted)" }}>click a step to rewind</span>
                  </div>
                )}
              </div>
              <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: "0 16px 16px", paddingBottom: 16 + regretHeight + wpaHeight }}>
                <LiveLog timeline={liveTimeline} onSeek={seekTo} busy={seekBusy} />
              </div>
            </div>
          </>
        ) : (
          // Replay mode's twin of the live split: the reasoning dock is its own
          // collapsible, resizable region and the event log gets an
          // independent scroller under it. They used to share ONE scroll box,
          // so a step with a wide search pushed the log out of sight and the
          // only way back was to scroll past every score row.
          <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
            {step && (
              <ReasoningDock
                title={null}
                collapsed={reasoningCollapsed}
                onToggle={() => setReasoningCollapsed(!reasoningCollapsed)}
                height={reasoningHeight}
                onResize={setReasoningHeight}
              >
                <Reasoning scores={step.scores} pv={step.pv} advice={step.meta?.advice}
                  heading={false} enemies={step.state.enemies} />
              </ReasoningDock>
            )}
            <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: 16, paddingBottom: 16 + regretHeight + wpaHeight }}>
              <EventLog entries={entries} current={idx} onPick={setIdx} />
            </div>
          </div>
        )}
      </div>
      <HelpOverlay open={helpOpen} onClose={() => setHelpOpen(false)} />
      {/* Outside the Tools menu on purpose: the menu unmounts its rows when it
          closes, which took the captured report's link with it. */}
      <BugReportToast result={bugResult} error={bugError}
        onDismiss={() => { setBugResult(null); setBugError(null); }} />
      <BridgeConnectDialog open={bridgeOpen} url={bridgeUrl}
        state={!bridgeUrl ? "off" : bridgeConnected ? "connected" : "unreachable"}
        error={bridgeProbeError} onChange={setBridgeUrl} onClose={() => setBridgeOpen(false)} />
      {loadOpen && (
        <SaveLoadDialog
          saves={saveList}
          loading={loadingSave}
          onLoad={loadSaved}
          onClose={() => setLoadOpen(false)}
        />
      )}
      {importOpen && (
        <ImportStateDialog
          onImported={afterImport}
          onClose={() => setImportOpen(false)}
          liveGame={bridgeConnected ? fetchLiveGameState : undefined}
          autoLive={importFromLive}
        />
      )}
      {treeOpen && (
        <BranchTree timeline={liveTimeline} onSeek={seekTo} busy={seekBusy}
          onClose={() => setTreeOpen(false)} />
      )}
      {bridgeTreeOpen && (
        <BranchTree timeline={bridgeTimeline} onSeek={bridgeSeekTo} busy={bridgeSeekBusy}
          onClose={() => setBridgeTreeOpen(false)} />
      )}
      {searchOpen && (
        <SearchTree tree={searchTree} busy={searchBusy} config={searchConfig}
          onRun={(c) => openSearchTree(c)}
          onClose={() => { setSearchOpen(false); setSearchFromBridge(false); }}
          potionNames={((mode === "play" ? live : step)?.state?.player?.potions ?? []).map((p) => p?.name)}
          enemies={(mode === "play" ? live : step)?.state?.enemies} />
      )}
      {mapOpen && (
        <MapOverlay
          rows={mode === "play" ? [] : mapRows}
          graphs={mode === "play" ? liveMapGraphs : mapGraphs}
          currentIdx={mode === "play" ? 0 : idx}
          loading={mapBusy}
          onJump={mode === "replay" ? (i) => { setIdx(Math.max(0, Math.min(count - 1, i))); setMapOpen(false); } : undefined}
          onClose={() => setMapOpen(false)}
          live={mode === "play"}
        />
      )}
      {deckOpen && (
        <PileViewer
          title="Deck"
          cards={((mode === "play" ? live : step)?.state?.player?.deck) ?? []}
          note="full deck"
          onClose={() => setDeckOpen(false)}
        />
      )}
      {analysisOpen && (
        <DeckAnalysis report={analysis} trend={trend}
          onSeek={mode === "replay" ? (i) => setIdx(Math.max(0, Math.min(count - 1, i))) : undefined}
          onClose={() => setAnalysisOpen(false)} />
      )}
      {regretOpen && (
        <RegretReview review={regret}
          live={regretLiveRef.current ?? false}
          onSeek={
            regretLiveRef.current
              // Live run: seek the live session to the decision's recorded
              // branch-tree node. Using the node id (not depth-on-active-path)
              // keeps later decisions clickable after an earlier seek has
              // rewound the active line — depth-only mapping would return null
              // for anything past the current position.
              ? (d) => {
                  const nodeId = liveSeekTargetForDecision(liveTimeline, d);
                  if (nodeId != null) void seekTo(nodeId);
                }
              : (d) => setIdx(Math.max(0, Math.min(count - 1, d.idx)))
          }
          onRerun={() => void runRegret(regretLiveRef.current ?? false)}
          onHeight={setRegretHeight}
          onClose={() => setRegretOpen(false)} />
      )}
      {wpaOpen && (
        <WpaReview review={wpa}
          live={wpaLiveRef.current ?? false}
          onSeek={
            wpaLiveRef.current
              // Live run: WPA points carry their branch-tree node id directly.
              ? (tp) => { if (tp.node_id != null) void seekTo(tp.node_id); }
              : (tp) => setIdx(Math.max(0, Math.min(count - 1, tp.idx)))
          }
          onVerify={
            wpaLiveRef.current
              ? async (tp) => {
                  if (tp.node_id == null) throw new Error("no node to verify");
                  const v = await api.live.wpaVerify(tp.node_id);
                  // The verify rollouts forked branches; refresh the tree view.
                  void refreshTimeline();
                  return v;
                }
              : undefined
          }
          onRerun={() => void runWpa(wpaLiveRef.current ?? false)}
          onHeight={setWpaHeight}
          onClose={() => setWpaOpen(false)} />
      )}
    </div>
    </SoundProvider>
    </EffectsProvider>
  );
}

/**
 * The side panel's "machine reasoning" dock: MCTS/advisor scores over the run
 * log, foldable and resizable.
 *
 * It shares the log's column, and a search over a big hand can fill it right
 * to its cap — which on a tablet in landscape leaves the log a couple of
 * lines. So it carries the same two affordances as the PPO dock under the
 * board: a caret that folds it to its header bar, and a grip on its bottom
 * edge that drags it to a height it remembers (see components/panelChrome.tsx).
 *
 * The `%` cap on the wrapper is the one that cannot be dragged away: whatever
 * px height is stored — from a desktop, from the other orientation — the dock
 * never takes more than 60% of the side panel, so the log is always on screen.
 */
function ReasoningDock({ title, collapsed, onToggle, height, onResize, children }: {
  /** Heading for the scores when they describe an earlier position (the
   *  "last search → …" case); a plain caption otherwise. */
  title: string | null;
  collapsed: boolean;
  onToggle: () => void;
  height: number;
  onResize: (h: number) => void;
  children: ReactNode;
}) {
  const bodyRef = useRef<HTMLDivElement>(null);
  return (
    <div style={{ flex: "0 1 auto", minHeight: 0, maxHeight: "60%", display: "flex",
      flexDirection: "column", overflow: "hidden", borderBottom: "1px solid var(--border)" }}>
      {/* The panel's caption lives here rather than inside `Reasoning` (which
          is told `heading={false}`): it has to stay on screen when the body is
          folded away, since it is what the caret hangs off. */}
      <div className="mono" style={{ flex: "0 0 auto", display: "flex", alignItems: "baseline",
        gap: 6, padding: "10px 16px 5px", fontSize: 10, color: "var(--muted)",
        borderBottom: collapsed ? undefined : "1px solid var(--border)", margin: "0 0 4px" }}>
        <CollapseButton collapsed={collapsed} onToggle={onToggle}
          label="the machine-reasoning panel" />
        <span style={{ flex: "0 0 auto", whiteSpace: "nowrap" }}>Machine reasoning</span>
        {title && (
          <span
            title="the search that chose this move — its rows score the position BEFORE it was played, not the board below"
            style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
          >
            · {title}
          </span>
        )}
      </div>
      {!collapsed && (
        <>
          <div ref={bodyRef} style={{ flex: "1 1 auto", minHeight: 0, maxHeight: height,
            overflowY: "auto", padding: "0 16px 8px" }}>
            {children}
          </div>
          <PanelGrip
            grow="down"
            label="drag to resize the reasoning panel"
            min={REASONING_MIN_H}
            max={REASONING_MAX_H}
            measure={() => bodyRef.current?.offsetHeight ?? height}
            onResize={onResize}
          />
        </>
      )}
    </div>
  );
}

function EffectsToggle() {
  const { toggle, setToggle } = useEffectsMode();
  const next = { full: "reduced", reduced: "off", off: "full" } as const;
  return (
    <button title="combat effects: full → reduced → off" onClick={() => setToggle(next[toggle])}>
      fx: {toggle}
    </button>
  );
}

// Deliberately two plain siblings, not a wrapping <span> — the surrounding
// toolbar (App.tsx's header `div`) is already a flex row with `gap: 8,
// alignItems: "center"`, which is what spaces every other control here
// (EffectsToggle, the hints/autobattle buttons). A wrapper adds a class with no
// matching CSS and no layout benefit, so the button and the volume slider
// join the row directly, styled like their neighbours (aria-pressed +
// accent-color-when-on, same as the 💡 hints button).
function SoundToggle() {
  const { on, setOn, volume, setVolume } = useSound();
  // Final review finding 11 (minor, 2026-08-23): sound mirrors the fx mode
  // (Ruling — see the design doc's §2.3), so `filterForMode` silences every
  // cue while `mode === "off"` — but the switch itself stays on and keeps
  // reading "🔊 sound ✓" the whole time, a state the spec's §2.4 says must
  // not exist (a switch that reads "on" while producing nothing). The
  // mirroring behaviour itself is correct and untouched here; only the
  // control's own display now says so, via a dimmed style + an explanatory
  // title (a native tooltip needs no new UI chrome).
  const { mode } = useEffectsMode();
  const silencedByFx = on && mode === "off";
  return (
    <>
      <button onClick={() => setOn(!on)} aria-pressed={on}
        title={silencedByFx
          ? "game sounds are on, but muted while combat effects (fx) are off"
          : "game sounds (on by default; the choice is remembered)"}
        style={{
          color: on ? "var(--accent-strong)" : undefined,
          opacity: silencedByFx ? 0.5 : undefined,
        }}>
        {on ? "🔊 sound ✓" : "🔇 sound"}
      </button>
      {on && (
        <input
          type="range" min={0.05} max={1} step={0.05} value={volume}
          aria-label="sound volume"
          onChange={(e) => setVolume(Number(e.target.value))}
        />
      )}
    </>
  );
}
