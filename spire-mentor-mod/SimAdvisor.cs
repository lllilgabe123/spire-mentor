// SimAdvisor.cs -- phase 6: the observe-and-advise loop.
//
// While SimRuntime.Mode == Advisor, a HUMAN plays the game and this class:
//   1. watches for decision points via PUBLIC game events (no Harmony):
//        - CombatManager.PlayerActionsDisabledChanged -- the exact "the
//          human may act now / no longer" edge (better than TurnStarted
//          alone: also covers mid-turn animation blocks),
//        - CombatManager.TurnStarted,
//        - CombatManager.History.Changed -- what did the human just do
//          (typed CardPlayFinishedEntry), which also means the board
//          changed and standing advice is stale,
//        - ActiveScreenContext.Updated -- screen opened/closed,
//   2. debounces them (events arrive in bursts while effects resolve),
//   3. extracts the SAME combat_action board the autopilot lane sends
//      (SimCombatHandler.BuildCombatState, one builder -- a second shape
//      would drift), wraps it as an `advice_request` (PROTOCOL.md v3),
//   4. sends it through BridgeServer's non-blocking advice lane, and
//   5. renders the correlated `advice` reply in SimHud -- unless the human
//      already moved (state_id mismatch / superseded request), in which
//      case it is dropped.
//
// Human actions are reported outbound as `human_action` (fire-and-forget)
// and recorded in the action log with the advice that was on screen.
//
// v1 scope: advice for COMBAT decisions. Non-combat screens (card reward,
// shop, rest, map) clear standing advice but do not request their own yet
// -- their payload builders live inside AutoSlayer's blocking handlers and
// need extraction first (the "per-screen advice" follow-up).
//
// The game is NOT paused while the human thinks: player choices suspend
// the action queue via TaskCompletionSource (GameAction.cs), SceneTree.
// Paused is never assigned -- so advice computes and renders while the
// screen sits open.
//
// Threading: game events fire on the main thread; advice replies arrive on
// the bridge's socket thread. Anything touching game state or the scene
// tree runs on the main thread (Callable.From(...).CallDeferred). Never
// throws into the game's event dispatch; never blocks it.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Actions;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal sealed class SimAdvisor
{
    public static readonly SimAdvisor Instance = new();

    /// <summary>
    /// True when nobody is watching this advisor session: the unattended
    /// harness flag (`advisor_autoplay` in SpireMentor.conf — default false,
    /// never persisted by the settings page, only ever written by the climb /
    /// F7 drivers). Human sessions keep the human pacing below; unattended
    /// ones take the fast values. Config-keyed (not autoplay-armed-keyed) on
    /// purpose: the pacing choice should be stable for a whole session, not
    /// flip with F7.
    /// </summary>
    private static bool Unattended => SimConfig.Instance.AdvisorAutoplay;

    /// <summary>
    /// Settle delay between a board/screen change and the advice request.
    /// 0.35 s is tuned for a human watching (batches the event bursts a
    /// single play emits). Unattended runs pay it on every one of ~500-900
    /// decisions, so they take a shorter one — the burst-batching still
    /// works at 0.1 s (the events arrive within a frame or two), and a
    /// mis-batched request only costs one extra request/cancel cycle.
    /// </summary>
    private static double DebounceSeconds => Unattended ? 0.10 : 0.35;

    /// <summary>
    /// How long to give an enqueued combat action to show up in the combat
    /// history before treating it as refused (see
    /// <see cref="WatchForRefusedAction" />). Long enough to sit out a card's
    /// animation; a false alarm only costs one extra advice request.
    /// </summary>
    private const double RefusedActionWatchdogSeconds = 5.0;

    /// <summary>How many refused recommendations in a row before autoplay ends
    /// the turn instead (see <see cref="OnActionRefused" />).</summary>
    private const int MaxRefusedBeforeEndTurn = 3;

    /// <summary>
    /// Cap on how long the settle barrier (<see cref="AwaitSettledThen" />)
    /// waits for the action queue to go quiet before snapshotting anyway.
    ///
    /// A backstop, not a timer: the barrier normally completes on the game's
    /// own <c>AfterActionExecuted</c> event. But a queue can be non-empty with
    /// NOTHING left to execute -- <c>ActionQueueSet.PauseActionForPlayerChoice</c>
    /// leaves the paused action at the front of its owner's queue -- and in
    /// that state no further event will ever fire. Waiting forever there would
    /// wedge the advisor, so the barrier gives up and proceeds, which is
    /// exactly the pre-barrier behaviour.
    /// </summary>
    private const double SettleTimeoutSeconds = 2.0;

    /// <summary>
    /// Identifies the barrier arming that is currently live. Monotonic, and
    /// deliberately NOT the advice generation: two <see cref="RequestAdviceNow" />
    /// calls can share a generation, and the second one's arming must retire
    /// the first one's backstop timer, which a generation comparison cannot
    /// see. Everything a barrier schedules -- its event handler, its backstop
    /// -- checks its own token and does nothing once that token is stale.
    /// </summary>
    private long _settleToken;

    /// <summary>The handler <see cref="AwaitSettledThen" /> subscribed, kept so
    /// it is detached exactly once.</summary>
    private Action<GameAction>? _settleHandler;

    private bool _subscribed;
    private long _generation;
    private string? _inflightRequestId;
    private long _inflightStateId = -1;

    // v5 coach correlation (PROTOCOL.md section 5, `human_action` v5): every
    // human action names the advice rendered for EXACTLY the position it was
    // taken in, or null. Three slots, all main-thread:
    //   _inflight*  : the request on the wire, tagged with the position epoch
    //                 (combat) or the screen signature it was built for;
    //   _shown*     : the accepted reply for the CURRENT position -- set in
    //                 OnAdviceReply, moved out by ScheduleAdvice (the position
    //                 changed);
    //   _actedFrom* : the advice for the position the player just LEFT.
    //                 ScheduleAdvice fires on the first event of a play
    //                 (actions disabled, the play's first history entries),
    //                 well before its CardPlayFinishedEntry is reported, so the
    //                 report reads this slot. Consumed by that report; cleared
    //                 whenever a new request goes out, because a position that
    //                 got its own request is not the one the player acted from.
    // A fast player's second card therefore finds both slots empty and
    // reports null -- never the first card's advice.
    private long _positionEpoch;
    private long _inflightEpoch = -1;
    private string _inflightSignature = "";
    private string? _shownRequestId;
    private long _shownStateId = -1;
    private bool _shownIsCombat;
    private bool _shownPeeked;
    private string? _actedFromRequestId;
    private long _actedFromStateId = -1;
    private bool _actedFromIsCombat;
    private bool _actedFromPeeked;

    /// <summary>
    /// The advisor's <c>capabilities.coach</c> from its latest reply; 0 until
    /// one arrives and whenever no advisor is connected. The v5
    /// <c>human_action</c> kinds an older advisor would mis-handle (end turn,
    /// non-combat choices) are sent only while this is &gt;= 1.
    /// </summary>
    private static int _advisorCoachCapability;
    public static int AdvisorCoachCapability => Volatile.Read(ref _advisorCoachCapability);

    /// <summary>The human pressed End turn (PlayerCmd.EndTurn prefix, see
    /// <c>HumanEndTurnPatch</c>); reported once the enemy side actually
    /// starts, so a backed-out end turn never reports.</summary>
    public static bool HumanEndTurnPending;

    /// <summary>Set around the mod's OWN PlayerCmd.EndTurn calls (takeover,
    /// refusal escape hatch, autopilot): those are not the human's.</summary>
    [ThreadStatic]
    public static bool SuppressEndTurnReport;

    /// <summary>In co-op the mod's end turn is an enqueued
    /// EndPlayerTurnAction, which reaches PlayerCmd.EndTurn later and with
    /// canBackOut: true, the human's signature; this marks it as the AI's.
    /// Only the main thread touches it (takeover and the queue both run
    /// there).</summary>
    private static bool _aiEndTurnQueued;

    /// <summary>F5 was pressed for the current position before its advice
    /// arrived: render the peek on the next accepted reply. Cleared whenever
    /// the position changes (ScheduleAdvice).</summary>
    private bool _peekLatched;

    private string? _lastBestLabel;
    private int _historySeen;
    private int _loggedFloor = -1;

    // Phase 7 takeover state. _lastBest is only ever non-null between a
    // fresh advice render and the next board change (every path that
    // schedules a new advice request clears it), so the hotkey can never
    // execute a recommendation against a board it wasn't computed for.
    private SimBestAction? _lastBest;

    /// <summary>
    /// The note that came with the standing recommendation, when the advisor
    /// sent one.
    ///
    /// It exists so the GAME LOG can tell a ranked pick from an arbitrary one.
    /// The advisor answers a surface no head covers by naming the first option
    /// and saying so in the note -- "defaulting to the first option, NOT a
    /// recommendation" (`realgame_advisor._unranked`) -- but that admission
    /// only ever reached the HUD and the advisor's own stdout. In `godot.log`,
    /// the line the climb's post-mortems actually read, a coin flip and a
    /// 4000-sim search printed the identical `Takeover executed on X: Y`. Eight
    /// of eight in-combat card selections were unranked before issue 5 was
    /// found, and nothing in any archived log said so.
    /// </summary>
    private string? _lastBestNote;

    /// <summary>The decision screen the standing recommendation belongs to,
    /// or null when the recommendation is a combat action.</summary>
    private SimScreenDecision? _lastScreen;

    /// <summary>Screen node the in-flight request was built from, so repeated
    /// events on the same screen don't cancel and restart it.</summary>
    private GodotObject? _inflightScreen;

    // Last decision the poll saw, so it only reacts to actual changes.
    private GodotObject? _polledScreen;
    private string _polledSignature = "";

    /// <summary>
    /// An advice request was abandoned because the advisor's socket was not up
    /// yet. The screen poll latches on (screen, signature), so without this the
    /// abandoned request is never re-issued and the screen stalls forever --
    /// see the retry in <see cref="PollScreen" />.
    /// </summary>
    private bool _adviceDeferredNoClient;

    /// <summary>Consecutive recommendations the live game refused; reset by any
    /// accepted action. See <see cref="OnActionRefused" />.</summary>
    private int _refusedInARow;

    /// <summary>Co-op autoplay: combat replies in a row that carried no
    /// move (the advisor failed on the position). See
    /// <see cref="OnNoMoveInCoop"/>.</summary>
    private int _noMoveInARow;

    /// <summary>Co-op autoplay: how long this round has already held moves
    /// for a teammate (<see cref="SimAdvice.WaitForAllyMs"/>), so two bots
    /// waiting on each other cost at most <see cref="AllyWaitBudgetMs"/> a
    /// round instead of deadlocking.</summary>
    private int _allyWaitRound = -1;
    private int _allyWaitUsedMs;
    private const int AllyWaitBudgetMs = 6000;

    /// <summary>Bumped by every <see cref="ScheduleAutoPlay"/>; a timer
    /// fires only if it is still the latest, so a long ally wait cannot be
    /// cut short by an older timer executing the newer advice.</summary>
    private int _autoPlayGen;

    /// <summary>Backing field — write through <see cref="SetAutoPlay"/> so
    /// SimRuntime.AdvisorAutoplayArmed (the speed patches' gate) stays in
    /// sync.</summary>
    private bool _autoPlay;
    private int _aiActionsPending;

    /// <summary>
    /// The decision autoplay last handed back (`pause_on`, GH #874), and the
    /// one the player resumed F7 on. Resuming on a handed-back screen means
    /// "you do it after all", so that decision -- every page and reveal of it
    /// -- must not pause again. See <see cref="PauseKey"/>.
    /// </summary>
    private string _pausedKey = "";
    private string _resumedKey = "";

    // Runaway guard. A screen choice that does not change the screen is not
    // idempotent in general: on the deck-select screens each repeat SELECTS
    // ANOTHER CARD, and a broken confirm turned that into 59 consecutive
    // "upgrade Stone Armor" clicks in one live run (2026-08-06). Whatever the
    // cause, going round the same decisions forever is never right.
    //
    // A WINDOW, not a consecutive-repeat counter. The first version of this
    // guard only noticed identical back-to-back choices and sailed straight
    // past the very next stall the harness found: claim card reward -> skip it
    // -> the loot screen re-offers it (skipping a card reward deliberately
    // does not retire it) -> claim -> skip, two ALTERNATING choices, neither
    // ever repeating. Counting occurrences inside a sliding window catches
    // cycles of any short period, which is the actual failure shape.
    private const int ChoiceWindow = 8;
    private const int MaxChoicesPerWindow = 3;
    private readonly Queue<string> _recentScreenChoices = new();

    /// <summary>Last screen choice autoplay executed and when, for the
    /// rapid-repeat dedupe in the takeover path (see the comment there).</summary>
    private string _lastChoiceKey = "";
    private ulong _lastChoiceAtMs;
    private const double ScreenPressCooldownSeconds = 2.0;

    /// <summary>Consecutive combat snapshots deferred by the unsettled-board
    /// guard in <see cref="RequestAdviceNow"/>; reset by any snapshot that is
    /// sent. Bounds the guard so a genuine zero-energy turn still gets
    /// advice.</summary>
    private int _unsettledRetries;

    private const int MaxUnsettledRetries = 8;

    /// <summary>Combat-history length when the current turn started, so the
    /// unsettled-board guard can tell "nothing has resolved yet this turn"
    /// (the race window) from a real spent-everything end-of-turn board.</summary>
    private int _turnStartHistoryMark = -1;

    /// <summary>A combat snapshot whose hand exists but has no playable card
    /// and no energy: at speed, that is the turn-start commands still
    /// resolving, not a real decision (see the guard's comment).</summary>
    private static bool LooksMidSettle(Dictionary<string, object> state)
    {
        if (state.TryGetValue("energy", out object? energyObj) &&
            energyObj is int energy && energy > 0)
        {
            return false;
        }
        if (state.TryGetValue("hand", out object? handObj) &&
            handObj is List<Dictionary<string, object?>> hand && hand.Count > 0)
        {
            foreach (Dictionary<string, object?> card in hand)
            {
                if (card.TryGetValue("playable", out object? playableObj) &&
                    playableObj is true)
                {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    /// <summary>Combat-history length when the loop window last recorded a
    /// choice. A different value means the board moved between then and now,
    /// so the window is not watching a loop. <c>-1</c> = not in combat, which
    /// is a stable value on purpose: it leaves the guard untouched out of
    /// combat.</summary>
    private int _choiceWindowHistoryMark = -1;

    private SimAdvisor()
    {
        SimRuntime.Changed += OnRuntimeChanged;
        BridgeServer.Instance.CoachReceived += OnCoachReceived;
    }

    /// <summary>Called from MainFile so the singleton (and its
    /// SimRuntime.Changed subscription) exists before the first mode flip.</summary>
    public void Prime() { }

    /// <summary>
    /// Called after an in-process run restore (SimRestore): the whole run
    /// scene was torn down and rebuilt, so any advice on screen and any
    /// armed best-action refer to a board that no longer exists. Clears
    /// them and (in Advisor mode) schedules fresh advice for the restored
    /// board. Safe from any thread.
    /// </summary>
    public void NotifyRunRestored()
    {
        RunOnMainThread(() =>
        {
            SimHud.Instance.ClearAdvice();
            // The torn-down screens must not read as skips, and no advice
            // from the abandoned branch may be attached to a later action.
            SimChoiceCapture.Reset();
            ForgetShownAdvice();
            HumanEndTurnPending = false;
            // A rewind abandons the branch the chips graded; the advisor's
            // withdrawal also reaches pinned ones.
            SimCoachChip.Instance.ClearUnpinned();
            SimCoachPanel.Instance.Hide();
            if (SimRuntime.Mode != SimMode.Advisor)
            {
                return;
            }
            EnsureSubscribed();
            ScheduleAdvice(); // bumps the generation: stale replies drop
        });
    }

    // ------------------------------------------------------------------
    // Mode + subscription lifecycle
    // ------------------------------------------------------------------

    private void OnRuntimeChanged()
    {
        RunOnMainThread(() =>
        {
            if (SimRuntime.Mode == SimMode.Advisor)
            {
                EnsureSubscribed();
                SimHud.Instance.SetAdvisorStatus(
                    BridgeServer.Instance.IsClientConnected ? SimAdvisorStatus.Connected : SimAdvisorStatus.None);
                SimAdvisorProcess.EnsureStarted();
                ScheduleAdvice();
            }
            else
            {
                SetAutoPlay(false);
                DropInflight();
                SimChoiceCapture.Reset();
                ForgetShownAdvice();
                HumanEndTurnPending = false;
                SimAdvisorProcess.Stop();
                SimHud.Instance.SetAdvisorStatus(SimAdvisorStatus.None);
                SimHud.Instance.ClearAdvice();
            }
        });
    }

    private void EnsureSubscribed()
    {
        if (_subscribed)
        {
            return;
        }
        // These singletons exist long before a human can flip the mode
        // (main menu is up); subscriptions stay installed for the session
        // and every handler self-gates on Mode == Advisor.
        CombatManager cm = CombatManager.Instance;
        cm.PlayerActionsDisabledChanged += _ => OnDecisionEdge();
        cm.TurnStarted += _ => OnTurnStarted();
        // CombatEnded(CombatRoom) fires with the room's combat state intact
        // (both game builds), so the round and room type are read from it.
        cm.CombatEnded += room => OnCombatEnded(room?.CombatState?.RoundNumber ?? 0, room?.RoomType.ToString() ?? "Combat");
        cm.PlayerUnendedTurn += _ =>
        {
            HumanEndTurnPending = false;
            _aiEndTurnQueued = false;
        };
        cm.History.Changed += OnHistoryChanged;
        ActiveScreenContext.Instance.Updated += OnScreenUpdated;
        StartScreenPoll();
        _subscribed = true;
        Logger.Log("[SimAdvisor] Observation hooks installed.");
    }

    /// <summary>
    /// Backstop for decision screens the event hooks never announce.
    /// `ActiveScreenContext.Updated` does not fire for everything pushed on
    /// NOverlayStack, and rooms like the loot screen, map, shop and events
    /// raise no combat events at all -- so autoplay simply stopped at the
    /// post-combat rewards (reported 2026-08-05) with nothing left to wake
    /// it. Polling the described decision's identity costs a scene-tree
    /// lookup twice a second and covers every present and future screen.
    /// </summary>
    private void StartScreenPoll()
    {
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return;
            }
            var timer = new Godot.Timer
            {
                Name = "SpireMentorAdvisorPoll",
                // The poll is the only wake-up for screens the event hooks
                // never announce (map, shop, rewards) — on those its period is
                // pure per-decision latency. Half it for unattended runs;
                // Describe() is a scene-tree lookup, cheap at 4 Hz.
                WaitTime = Unattended ? 0.25 : 0.5,
                Autostart = true,
                OneShot = false,
            };
            timer.Connect(Godot.Timer.SignalName.Timeout, Callable.From(PollScreen));
            tree.Root.AddChildSafely(timer);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Screen poll unavailable: {ex.Message}");
        }
    }

    private void PollScreen()
    {
        if (SimRuntime.Mode != SimMode.Advisor)
        {
            return;
        }
        try
        {
            SimScreenDecision? current = SimScreenAdvice.Describe();
            GodotObject? screen = current?.Screen;
            // Node identity is NOT enough to tell one decision from the next:
            // the map, the shop and the event room are the run's own
            // long-lived nodes, so every map choice in an act arrives on the
            // same object -- and with the same option count often enough that
            // keying on the pair silently answered the first map of the act
            // and then nothing (2026-08-05). The signature is what actually
            // differs (where we stand, what is on offer, what it costs).
            string signature = current?.SignatureOrDefault ?? "";
            bool unchanged = ReferenceEquals(screen, _polledScreen) && signature == _polledSignature;
            // Every tick, changed or not: a decision that just closed waits a
            // few ticks for its pick to land in the deck / relics / map.
            SimChoiceCapture.OnPoll(current, changed: !unchanged);
            // The recap / autopsy panel lives as long as its screens do.
            SimCoachPanel.Instance.OnScreen(current?.AdviceFor);
            if (unchanged)
            {
                // ... but "unchanged" must not mean "nothing to do". An advice
                // request issued before the advisor process finished
                // connecting is dropped on the floor, and this latch means
                // nothing ever asks again: on 2026-08-06 the Neow screen came
                // up in the ~1 s between `advisor_autoplay` arming autoplay and
                // the Python advisor's socket arriving, and the run sat on the
                // first decision of the game indefinitely. The same hole swallows
                // any advisor restart mid-run.
                if (_adviceDeferredNoClient && BridgeServer.Instance.IsClientConnected)
                {
                    _adviceDeferredNoClient = false;
                    Logger.Log("[SimAdvisor] Advisor connected; re-requesting advice for the current decision.");
                    ScheduleAdvice();
                }
                return;
            }
            _polledScreen = screen;
            _polledSignature = signature;
            if (current != null)
            {
                Logger.Log($"[SimAdvisor] Decision up: {current.ScreenName} [{signature}]");
            }
            ScheduleAdvice();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Screen poll failed: {ex.Message}");
        }
    }

    // ------------------------------------------------------------------
    // Event handlers (main thread)
    // ------------------------------------------------------------------

    private void OnDecisionEdge()
    {
        if (SimRuntime.Mode != SimMode.Advisor)
        {
            return;
        }
        ScheduleAdvice();
    }

    /// <summary>Turn boundary: a log marker with the round number and the
    /// player's HP/block, then the usual advice refresh.</summary>
    private void OnTurnStarted()
    {
        if (SimRuntime.Mode != SimMode.Advisor)
        {
            return;
        }
        bool enemySide = false;
        try
        {
            enemySide = CombatManager.Instance.IsEnemyTurnStarted;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] IsEnemyTurnStarted: {ex.Message}");
        }
        if (enemySide)
        {
            // The enemy side has started: the human's end turn is now final
            // (a canBackOut cancel can no longer happen). Report it here, not
            // in the PlayerCmd.EndTurn prefix.
            if (HumanEndTurnPending)
            {
                HumanEndTurnPending = false;
                SendEndTurn();
            }
        }
        else
        {
            HumanEndTurnPending = false;
            // The player's turn in a fight: the previous fight's recap is over.
            SimCoachPanel.Instance.OnCombatTurn();
        }
        _aiEndTurnQueued = false;
        // A streak is about one position: no-move replies from an earlier
        // turn or fight must not end this turn on its first miss.
        _noMoveInARow = 0;
        _turnStartHistoryMark = CombatHistoryMark();
        _unsettledRetries = 0;
        try
        {
            RunState? runState = RunManager.Instance.DebugOnlyGetState();
            var creature = LocalContext.GetMe(runState)?.Creature;
            int round = CombatManager.Instance.DebugOnlyGetState()?.RoundNumber ?? 0;
            string hp = creature != null ? $"  HP {creature.CurrentHp}/{creature.MaxHp}" : "";
            string block = creature != null && creature.Block > 0 ? $"  block {creature.Block}" : "";
            SimActionLog.Record(new SimActionLogEntry
            {
                Actor = "game",
                Act = (runState?.CurrentActIndex ?? -1) + 1,
                Floor = runState?.TotalFloor ?? 0,
                Action = $"— turn {round} —{hp}{block}",
            });
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] OnTurnStarted log failed: {ex.Message}");
        }
        ScheduleAdvice();
    }

    private void OnCombatEnded(int round, string roomType)
    {
        if (SimRuntime.Mode != SimMode.Advisor)
        {
            return;
        }
        ReportCombatEnded(round, roomType);
        DropInflight();
        SimHud.Instance.ClearAdvice();
        HumanEndTurnPending = false;
        ForgetShownAdvice();
        SimCoachChip.Instance.ClearUnpinned();
        _historySeen = 0;
        try
        {
            RunState? runState = RunManager.Instance.DebugOnlyGetState();
            var creature = LocalContext.GetMe(runState)?.Creature;
            SimActionLog.Record(new SimActionLogEntry
            {
                Actor = "game",
                Act = (runState?.CurrentActIndex ?? -1) + 1,
                Floor = runState?.TotalFloor ?? 0,
                Action = creature != null
                    ? $"— combat ended —  HP {creature.CurrentHp}/{creature.MaxHp}"
                    : "— combat ended —",
            });
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] OnCombatEnded log failed: {ex.Message}");
        }
    }

    /// <summary>`combat_ended` to a coach-2 advisor (PROTOCOL.md section 3):
    /// the advisor answers a won fight with a recap.</summary>
    private static void ReportCombatEnded(int round, string roomType)
    {
        if (AdvisorCoachCapability < SimCoachReports.RequiredCoachCapability ||
            !BridgeServer.Instance.IsClientConnected)
        {
            return;
        }
        try
        {
            RunState? runState = RunManager.Instance.DebugOnlyGetState();
            var me = LocalContext.GetMe(runState);
            var creature = me?.Creature;
            if (creature == null)
            {
                return;
            }
            var msg = SimCoachReports.CombatEnded(
                victory: creature.CurrentHp > 0, round, creature.CurrentHp, creature.MaxHp, roomType);
            SimStateExtractor.ApplyEnvelope(msg, runState, me);
            BridgeServer.Instance.SendState(JsonSerializer.Serialize(msg));
            Logger.Log($"[SimAdvisor] combat_ended sent (round {round}, HP {creature.CurrentHp}/{creature.MaxHp}).");
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] combat_ended report failed: {ex.Message}");
        }
    }

    /// <summary>Advisor-mode `game_over` to a coach-2 advisor, once per run
    /// (the latch is shared with the autopilot's send): the advisor answers
    /// a death with the autopsy.</summary>
    private static void ReportGameOver()
    {
        if (AdvisorCoachCapability < SimCoachReports.RequiredCoachCapability ||
            !BridgeServer.Instance.IsClientConnected)
        {
            return;
        }
        try
        {
            RunState? runState = RunManager.Instance.DebugOnlyGetState();
            if (!SimCoachReports.TryClaimGameOver(runState))
            {
                return;
            }
            bool victory = SimGameOverScreenHandler.ReadVictory();
            var msg = SimCoachReports.GameOver(victory);
            SimStateExtractor.ApplyEnvelope(msg, runState, LocalContext.GetMe(runState));
            BridgeServer.Instance.SendState(JsonSerializer.Serialize(msg));
            Logger.Log($"[SimAdvisor] game_over sent ({(victory ? "victory" : "defeat")}).");
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] game_over report failed: {ex.Message}");
        }
    }

    private void OnScreenUpdated()
    {
        if (SimRuntime.Mode != SimMode.Advisor)
        {
            return;
        }
        try
        {
            LogFloorChange();
            // A screen decision (card reward, card select, loot) or a combat
            // board -- ScheduleAdvice picks whichever applies. Screens that
            // are not decisions just produce no request and clear the HUD.
            ScheduleAdvice();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] OnScreenUpdated: {ex.Message}");
        }
    }

    /// <summary>One log line per floor entered, with the room type, HP and
    /// gold -- the spine the rest of the log hangs off.</summary>
    private void LogFloorChange()
    {
        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        if (runState == null)
        {
            return;
        }
        int floor = runState.TotalFloor;
        if (floor == _loggedFloor)
        {
            return;
        }
        _loggedFloor = floor;

        Player? me = LocalContext.GetMe(runState);
        var creature = me?.Creature;
        string room = "";
        try
        {
            room = runState.CurrentRoom?.RoomType.ToString() ?? "";
        }
        catch
        {
            // Between rooms; the floor marker alone is still useful.
        }
        string hp = creature != null ? $"  HP {creature.CurrentHp}/{creature.MaxHp}" : "";
        string gold = me != null ? $"  gold {me.Gold}" : "";
        SimActionLog.Record(new SimActionLogEntry
        {
            Actor = "game",
            Act = runState.CurrentActIndex + 1,
            Floor = floor,
            Action = $"════ floor {floor}{(room.Length > 0 ? $" · {room}" : "")}{hp}{gold}",
        });
    }

    private void OnHistoryChanged()
    {
        if (SimRuntime.Mode != SimMode.Advisor)
        {
            return;
        }
        try
        {
            ReportNewHumanActions();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] OnHistoryChanged: {ex.Message}");
        }
        ScheduleAdvice();
    }

    // ------------------------------------------------------------------
    // Debounce + request
    // ------------------------------------------------------------------

    private void ScheduleAdvice()
    {
        _lastBest = null; // board is (about to be) different; stale recommendation must not execute
        _lastScreen = null;
        // A peek is for one decision only: it hides once that is made.
        _peekLatched = false;
        SimHud.Instance.ClearPeek();
        // The position changed: the advice on screen now belongs to the
        // position the player LEFT (see the correlation slots above).
        _positionEpoch++;
        if (_shownRequestId != null)
        {
            _actedFromRequestId = _shownRequestId;
            _actedFromStateId = _shownStateId;
            _actedFromIsCombat = _shownIsCombat;
            _actedFromPeeked = _shownPeeked;
            _shownRequestId = null;
            _shownStateId = -1;
            _shownPeeked = false;
        }
        long gen = ++_generation;
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return;
            }
            SceneTreeTimer timer = tree.CreateTimer(DebounceSeconds);
            timer.Connect(SceneTreeTimer.SignalName.Timeout, Callable.From(() =>
            {
                if (gen == _generation)
                {
                    RequestAdviceNow();
                }
            }));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] ScheduleAdvice: {ex.Message}");
        }
    }

    /// <summary>
    /// Is the action queue quiet enough to snapshot? Call ONLY from inside
    /// <c>AfterActionExecuted</c>.
    ///
    /// <c>GetReadyAction</c> answers "will anything run on its own?", and its
    /// selection rules are exactly the ones we want: a queue awaiting a player
    /// choice is SKIPPED rather than waited on, so a parked card-select reads
    /// as quiet instead of blocking us forever. <c>null</c> therefore means
    /// the board has stopped moving by itself -- either nothing is queued, or
    /// everything left is waiting on the player, which IS the decision point.
    ///
    /// The second clause mirrors <c>CombatManager</c>'s own settle helper
    /// (<c>CombatManager.cs:1132</c>): a ready action that is not player-driven
    /// (a hook action, ReadyToBeginEnemyTurn) does not invalidate the snapshot.
    ///
    /// Its doc warns "should not be called when an action is already being
    /// executed" and it THROWS on an unexpected front-of-queue state, which is
    /// why this is event-scoped rather than polled. Wrapped anyway: a throw
    /// here must degrade to "not settled yet", never take down the advisor.
    /// </summary>
    private static bool QueueIsQuiet()
    {
        try
        {
            GameAction? ready = RunManager.Instance.ActionQueueSet.GetReadyAction();
            return ready == null || !ActionQueueSet.IsGameActionPlayerDriven(ready);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] GetReadyAction while settling: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Run <paramref name="then" /> once the game has stopped moving the board
    /// on its own, or after <see cref="SettleTimeoutSeconds" />.
    ///
    /// This is the fix for a race the 0.10 s debounce only papered over. The
    /// mod enqueues its plays through <c>ActionQueueSynchronizer</c>, so its
    /// LAST play can still be mid-resolution when the next advice request is
    /// built. <c>PlayerActionsDisabled</c> can be false while that is true, and
    /// the generation counter does not help either: it only bumps on
    /// <c>History.Changed</c>, and an action finishing its own resolution is
    /// not a new history event. The snapshot then shows energy the game has
    /// already spent, the search picks a card for it, and the pre-flight
    /// <c>CanPlay</c> check refuses it -- the <c>energy=0</c> signature that is
    /// the largest refusal class in every graded batch (17 in check18, 18 in
    /// check19).
    ///
    /// Waiting on the game's own event instead of a duration is what the game
    /// does internally, and it is the same lesson as the 2026-08-10 speed
    /// patches: the delays that were removed had been doing hidden
    /// synchronisation, and four latent defects surfaced at once when they
    /// went. This replaces the last of that incidental timing with a fact.
    ///
    /// NOT applied to the screen lane. A screen is asked BEFORE this in
    /// <see cref="RequestAdviceNow" />, and a screen is open precisely because
    /// an action parked for player choice -- so the queue is legitimately
    /// non-empty and there is nothing to wait for.
    /// </summary>
    private void AwaitSettledThen(long gen, Action<bool> then)
    {
        // Already quiet: the common case, and it must not cost a frame.
        // `IsEmpty` is a plain property read with no precondition, unlike
        // `GetReadyAction`, so it is the only probe safe to make from here.
        try
        {
            if (RunManager.Instance.ActionQueueSet.IsEmpty)
            {
                then(false);
                return;
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] settle probe: {ex.Message}");
            then(false);
            return;
        }

        DetachSettleHandler();
        long token = ++_settleToken;
        int waited = 0;

        void OnExecuted(GameAction _)
        {
            // A later arming replaced us. Unsubscribe THIS delegate by name
            // rather than going through DetachSettleHandler, which would tear
            // down the barrier that superseded us.
            if (token != _settleToken)
            {
                Unsubscribe(OnExecuted);
                return;
            }
            // A newer ScheduleAdvice superseded us; that generation owns the
            // next request and this one must simply stop.
            if (gen != _generation)
            {
                DetachSettleHandler();
                return;
            }
            waited++;
            if (!QueueIsQuiet())
            {
                return;
            }
            DetachSettleHandler();
            Logger.Log(
                $"[SimAdvisor] settle barrier: queue quiet after {waited} action(s).");
            CompleteSettle(gen, () => then(true));
        }

        _settleHandler = OnExecuted;
        try
        {
            RunManager.Instance.ActionExecutor.AfterActionExecuted += OnExecuted;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] could not arm the settle barrier: {ex.Message}");
            _settleHandler = null;
            then(false);
            return;
        }

        // Backstop for the queue that is non-empty but has nothing left to
        // run (every action parked for player choice): no event is coming.
        //
        // `ignoreTimeScale` because this measures "how long am I prepared to
        // be stuck", which is a real-world quantity. The mod drives the game
        // at a raised Engine.TimeScale and NHitStop stomps that outright, so a
        // scaled timer would be a different backstop on every board: far too
        // short under the speed patches -- firing mid-resolution, which is the
        // bug the barrier exists to fix -- and unbounded at a time scale of 0.
        if (Engine.GetMainLoop() is SceneTree tree)
        {
            SceneTreeTimer bail = tree.CreateTimer(
                SettleTimeoutSeconds,
                processAlways: true,
                processInPhysics: false,
                ignoreTimeScale: true);
            bail.Connect(SceneTreeTimer.SignalName.Timeout, Callable.From(() =>
            {
                if (token != _settleToken || _settleHandler == null)
                {
                    return;
                }
                if (gen != _generation)
                {
                    DetachSettleHandler();
                    return;
                }
                Logger.Log(
                    "[SimAdvisor] settle barrier timed out; snapshotting anyway.");
                DetachSettleHandler();
                CompleteSettle(gen, () => then(true));
            }));
        }
    }

    /// <summary>
    /// Hand <paramref name="then" /> back to the engine OUTSIDE the callstack
    /// that completed the barrier.
    ///
    /// <c>AfterActionExecuted</c> fires from inside <c>ActionExecutor</c>'s own
    /// frame, with the action it just ran still on the stack. <paramref
    /// name="then" /> is not a small callback: it walks the whole combat
    /// state, serialises it, writes it to a socket, and on the far side can
    /// schedule a play back into the very queue we are standing in. Running
    /// that re-entrantly is the shape of the defects the 2026-08-10 speed
    /// patches exposed, where removing incidental delays let the mod re-enter
    /// the game mid-resolution. The pre-barrier build always reached here from
    /// a SceneTreeTimer -- idle context -- and CallDeferred restores that
    /// property without putting a duration back into the design.
    ///
    /// Deferring means the board can move again before we run, so the
    /// generation is re-checked on the far side of the frame boundary.
    /// </summary>
    private void CompleteSettle(long gen, Action then)
    {
        try
        {
            Callable.From(() =>
            {
                if (gen != _generation)
                {
                    return;
                }
                Guarded(then);
            }).CallDeferred();
        }
        catch (Exception ex)
        {
            // Could not defer. Better a re-entrant snapshot than none at all:
            // no advice at all stalls autoplay for the rest of the turn.
            Logger.Log($"[SimAdvisor] could not defer the settled snapshot: {ex.Message}");
            Guarded(then);
        }
    }

    /// <summary>Detach one specific settle handler, without assuming it is the
    /// one this advisor currently owns.</summary>
    private static void Unsubscribe(Action<GameAction> handler)
    {
        try
        {
            RunManager.Instance.ActionExecutor.AfterActionExecuted -= handler;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] detaching the settle barrier: {ex.Message}");
        }
    }

    /// <summary>Unsubscribe the settle handler if one is armed. Idempotent:
    /// both the event path and the timeout path call it.</summary>
    private void DetachSettleHandler()
    {
        if (_settleHandler == null)
        {
            return;
        }
        Unsubscribe(_settleHandler);
        _settleHandler = null;
    }

    private void RequestAdviceNow()
    {
        try
        {
            if (SimRuntime.Mode != SimMode.Advisor)
            {
                return;
            }
            // A screen decision outranks the combat board: while a card
            // reward / card select / loot screen is up, THAT is what the
            // player is being asked, even mid-combat (Armaments).
            SimScreenDecision? screen = SimScreenAdvice.Describe();
            if (screen != null)
            {
                RequestScreenAdvice(screen);
                return;
            }

            if (!BridgeServer.Instance.IsClientConnected)
            {
                SimHud.Instance.SetAdvisorStatus(SimAdvisorStatus.None);
                _adviceDeferredNoClient = true;
                Volatile.Write(ref _advisorCoachCapability, 0);
                return;
            }

            // Everything below reads the BOARD, so it waits until the game
            // has stopped changing it. See `AwaitSettledThen`; the guards are
            // deliberately re-checked on the far side, because the whole
            // premise is that the board moved while we waited.
            AwaitSettledThen(_generation, RequestCombatAdviceNow);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] RequestAdviceNow: {ex.Message}");
        }
    }

    /// <summary>
    /// The combat half of <see cref="RequestAdviceNow" />, re-entered once the
    /// action queue has gone quiet. Split out so every board-dependent guard
    /// below is evaluated against the SETTLED board rather than the one that
    /// happened to be up when the debounce fired.
    /// </summary>
    private void RequestCombatAdviceNow(bool boardMayHaveMoved)
    {
        try
        {
            if (SimRuntime.Mode != SimMode.Advisor)
            {
                return;
            }
            // Re-asked, not assumed: settling can take us past the end of the
            // fight, or onto a screen that outranks the board.
            //
            // Only when we actually waited, though. When the queue was
            // already quiet, the answer RequestAdviceNow got moments ago is
            // still the answer.
            if (boardMayHaveMoved)
            {
                SimScreenDecision? nowScreen = SimScreenAdvice.Describe();
                if (nowScreen != null)
                {
                    RequestScreenAdvice(nowScreen);
                    return;
                }
            }

            CombatManager cm = CombatManager.Instance;
            if (!cm.IsInProgress || cm.PlayerActionsDisabled)
            {
                // Not a decision point; keep whatever advice is showing
                // (mid-animation) -- staleness is handled by the events
                // that fire when the board actually changes.
                return;
            }

            RunState? runState = RunManager.Instance.DebugOnlyGetState();
            Player? player = LocalContext.GetMe(runState);
            if (player == null)
            {
                return;
            }
            // Not the play phase yet: no snapshot. `StartTurn` drops
            // PlayerActionsDisabled at the very START of turn setup, before
            // the energy reset and the hand draw (CombatManager.cs:465 vs
            // SetupPlayerTurn), so the decision edge above fires on a board
            // with an empty hand. The search can only answer "End turn" to
            // that, and autoplay executed it: whole turns ended card-less
            // with a full hand about to be drawn (2026-10-16, Fuzzy Wurm
            // rounds 3-5). TurnStarted fires on the switch to Play and
            // re-requests, so nothing is lost by waiting for it. Same gate
            // as the autopilot lane (SimCombatHandler.IsPlayerInPlayPhase).
            if (player.PlayerCombatState?.Phase != PlayerTurnPhase.Play)
            {
                return;
            }

            Dictionary<string, object>? state = SimCombatHandler.BuildCombatState(player);
            if (state == null)
            {
                return;
            }
            // Advisor-lane only (NOT BuildCombatState, which the autopilot
            // wire shares): the turn number lets the workbench tree map each
            // combat decision to an undo_turn target.
            int turnNumber = player.PlayerCombatState?.TurnNumber ?? 0;
            if (turnNumber > 0)
            {
                state["turn"] = turnNumber;
            }

            // Unsettled-board guard. The "actions enabled" edge can fire
            // while the turn-start draw/energy commands are still resolving;
            // a snapshot taken inside that window shows a non-empty hand
            // with zero energy and nothing playable, the client's
            // wire-legality veto then strips every card, and the resulting
            // advice degenerates to "End turn" — which autoplay executes,
            // silently skipping the whole turn. At human pacing the 0.35 s
            // debounce mostly outlived the window; at the unattended pacing
            // it does not: smoke run 3 (2026-08-10) lost a 999-HP run to
            // Bowlbugs by ending EVERY turn card-less, one race at a time.
            // Such a board is not a decision point yet — re-ask shortly.
            // Bounded: a turn that GENUINELY starts at zero energy (a
            // debuff, not a race) must still get an answer, so after
            // MaxUnsettledRetries the snapshot is sent as-is.
            // Second clause: a hand full of unplayable cards at zero energy
            // is also the NORMAL end-of-turn shape (spent everything, junk
            // left over). What separates the race from that is history: at a
            // real end-of-turn decision this turn's plays have already
            // appended entries, while in the race window nothing has
            // resolved since TurnStarted. Only defer the latter.
            if (LooksMidSettle(state) &&
                CombatHistoryMark() == _turnStartHistoryMark &&
                _unsettledRetries < MaxUnsettledRetries)
            {
                _unsettledRetries++;
                if (Engine.GetMainLoop() is SceneTree settleTree)
                {
                    SceneTreeTimer settle = settleTree.CreateTimer(0.15);
                    settle.Connect(SceneTreeTimer.SignalName.Timeout, Callable.From(() =>
                    {
                        if (SimRuntime.Mode == SimMode.Advisor)
                        {
                            ScheduleAdvice();
                        }
                    }));
                }
                return;
            }
            _unsettledRetries = 0;

            long stateId = state.TryGetValue("state_id", out object? sidObj) && sidObj is long sid ? sid : -1;
            state["type"] = "advice_request";
            state["advice_for"] = "combat_action";
            // The game's own combat log since the last decision -- the rows the
            // workbench's log panel renders for a live game, in the same shape
            // the Rust engine emits for a simulated one. Display only; the
            // advisor ignores it.
            WatchPlayerChoices();
            state["combat_log"] = SimCombatLog.Drain();
            state["capabilities"] = ModCapabilities();

            DropInflight();

            string json;
            try
            {
                json = JsonSerializer.Serialize(state);
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimAdvisor] Could not serialize advice_request: {ex.Message}");
                return;
            }

            string? requestId = BridgeServer.Instance.SendAdviceRequest(json, OnAdviceReply);
            if (requestId == null)
            {
                SimHud.Instance.SetAdvisorStatus(SimAdvisorStatus.None);
                _adviceDeferredNoClient = true;
                return;
            }
            _adviceDeferredNoClient = false;
            _inflightRequestId = requestId;
            _inflightStateId = stateId;
            _inflightEpoch = _positionEpoch;
            _inflightSignature = "";
            ClearActedFrom();
            UpdateThinkFps();
            SimHud.Instance.SetAdvisorStatus(SimAdvisorStatus.Thinking);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] RequestAdviceNow: {ex.Message}");
        }
    }

    /// <summary>
    /// Ask about (or locally answer) a decision screen the human is looking
    /// at. Screens are addressed by option index, so the recommendation is a
    /// `choose` action the mod executes by clicking the screen.
    /// </summary>
    private void RequestScreenAdvice(SimScreenDecision screen)
    {
        _lastScreen = screen;

        // A screen decision does not change while its screen is open, but
        // combat animations keep firing History.Changed underneath it -- and
        // each of those used to cancel the in-flight request and start over.
        // With a cold PPO net (45 s to load torch + the checkpoint on the
        // first card reward, measured 2026-08-05) the reply ALWAYS arrived
        // after a cancellation, so a card reward screen showed nothing at
        // all. Let a request for the same screen node run to completion.
        if (!screen.Local && _inflightRequestId != null &&
            _inflightScreen != null && ReferenceEquals(_inflightScreen, screen.Screen))
        {
            return;
        }

        if (screen.AdviceFor == "game_over")
        {
            ReportGameOver();
        }

        if (screen.AdviceFor == "game_over" && _autoPlay)
        {
            // The run is over: there is nothing left to play, and clicking on
            // through the summary is the player's call, not ours.
            SetAutoPlay(false);
            Logger.Log("[SimAdvisor] Auto-play off: the run has ended.");
            SimActionLog.Note("Auto-play off — run over");
        }

        if (screen.Local)
        {
            DropInflight();
            _lastBest = screen.LocalAdvice?.Best;
            _lastBestLabel = _lastBest?.Label;
            _lastBestNote = screen.LocalAdvice?.Note;
            SimHud.Instance.SetAdvisorStatus(SimAdvisorStatus.Connected);
            if (screen.LocalAdvice != null)
            {
                SimHud.Instance.ShowAdvice(screen.LocalAdvice, null);
                RenderLatchedPeek();
            }
            if (_autoPlay && !PauseForPlayer(screen, screen.HandBack) && _lastBest != null)
            {
                ScheduleAutoPlay();
            }
            return;
        }

        if (!BridgeServer.Instance.IsClientConnected)
        {
            SimHud.Instance.SetAdvisorStatus(SimAdvisorStatus.None);
            _adviceDeferredNoClient = true;
            Volatile.Write(ref _advisorCoachCapability, 0);
            return;
        }

        long stateId = screen.State.TryGetValue("state_id", out object? sidObj) && sidObj is long sid ? sid : -1;
        var payload = new Dictionary<string, object>(screen.State)
        {
            ["type"] = "advice_request",
            ["advice_for"] = screen.AdviceFor,
            // Drain here too: a fight's LAST rows (the killing blow, the
            // end-of-combat powers) land after its final combat decision, so
            // without this they would sit unsent until the next fight.
            ["combat_log"] = SimCombatLog.Drain(),
            ["capabilities"] = ModCapabilities(),
        };

        DropInflight();

        string json;
        try
        {
            json = JsonSerializer.Serialize(payload);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Could not serialize {screen.AdviceFor} advice_request: {ex.Message}");
            return;
        }

        string? requestId = BridgeServer.Instance.SendAdviceRequest(json, OnAdviceReply);
        if (requestId == null)
        {
            SimHud.Instance.SetAdvisorStatus(SimAdvisorStatus.None);
            _adviceDeferredNoClient = true;
            return;
        }
        _adviceDeferredNoClient = false;
        _inflightRequestId = requestId;
        _inflightStateId = stateId;
        _inflightScreen = screen.Screen;
        _inflightEpoch = _positionEpoch;
        _inflightSignature = screen.SignatureOrDefault;
        ClearActedFrom();
        UpdateThinkFps();
        SimHud.Instance.SetAdvisorStatus(SimAdvisorStatus.Thinking);
    }

    private void DropInflight()
    {
        string? previous = _inflightRequestId;
        _inflightRequestId = null;
        _inflightStateId = -1;
        _inflightScreen = null;
        _inflightEpoch = -1;
        _inflightSignature = "";
        UpdateThinkFps();
        if (previous != null)
        {
            BridgeServer.Instance.CancelAdviceRequest(previous);
        }
    }

    /// <summary>
    /// Keep the SimThinkFps cap aligned with "an unattended think is in
    /// flight". Main thread only (all _inflightRequestId transitions are).
    /// The attended advisor session never enters: a human is watching the
    /// screen precisely while the advice computes.
    /// </summary>
    private bool _thinkFpsHeld;

    private void UpdateThinkFps()
    {
        bool want = Unattended && _inflightRequestId != null;
        if (want == _thinkFpsHeld)
        {
            return;
        }
        _thinkFpsHeld = want;
        if (want)
        {
            SimThinkFps.Begin();
        }
        else
        {
            SimThinkFps.End();
        }
    }

    // ------------------------------------------------------------------
    // Advice reply (socket thread -> main thread)
    // ------------------------------------------------------------------

    private void OnAdviceReply(JsonElement root)
    {
        // Socket thread: only immutable parsing here, render deferred.
        try
        {
            string? requestId = root.TryGetProperty("request_id", out JsonElement idEl) ? idEl.GetString() : null;
            long stateId = root.TryGetProperty("state_id", out JsonElement sEl) &&
                           sEl.ValueKind == JsonValueKind.Number && sEl.TryGetInt64(out long s)
                ? s
                : -1;

            SimAdvice advice = root.TryGetProperty("advice", out JsonElement adviceEl)
                ? SimAdvice.Parse(adviceEl)
                : new SimAdvice();
            // Every reply speaks for the advisor build that sent it, accepted
            // or not: absent = an advisor without coach support.
            Volatile.Write(ref _advisorCoachCapability, CoachCapabilityOf(root));

            RunOnMainThread(() =>
            {
                if (SimRuntime.Mode != SimMode.Advisor)
                {
                    return;
                }
                if (requestId != _inflightRequestId || (stateId >= 0 && stateId != _inflightStateId))
                {
                    Logger.Log($"[SimAdvisor] Dropping stale advice (request {requestId}, state {stateId}).");
                    return;
                }
                // The correlation fix: the state id is retired WITH the request
                // id (it used to linger and mislabel later human actions).
                bool isCombat = _inflightScreen == null;
                GodotObject? forScreen = _inflightScreen;
                string forSignature = _inflightSignature;
                // The request's own state id: a reply that omits `state_id`
                // still names the position it was asked for.
                long forStateId = stateId >= 0 ? stateId : _inflightStateId;
                bool samePosition = isCombat
                    ? _inflightEpoch == _positionEpoch
                    : ReferenceEquals(forScreen, _polledScreen) && forSignature == _polledSignature;
                _inflightRequestId = null;
                _inflightStateId = -1;
                _inflightScreen = null;
                _inflightEpoch = -1;
                _inflightSignature = "";
                UpdateThinkFps();

                // Only advice for the position still on screen may grade the
                // next action; a reply that raced a board change renders as
                // before but is never attached to a human action.
                if (samePosition && requestId != null)
                {
                    _shownRequestId = requestId;
                    _shownStateId = forStateId;
                    _shownIsCombat = isCombat;
                    _shownPeeked = false;
                    if (!isCombat)
                    {
                        SimChoiceCapture.OnAdviceShown(forScreen, forSignature, requestId, forStateId);
                    }
                }

                _lastBestLabel = advice.Options.FirstOrDefault(o => o.IsBest)?.Label
                                 ?? advice.Options.FirstOrDefault()?.Label;
                _lastBest = advice.Best;
                _lastBestNote = advice.Note;

                SimHud.Instance.SetAdvisorStatus(SimAdvisorStatus.Connected);
                RunState? runState = RunManager.Instance.DebugOnlyGetState();
                SimHud.Instance.ShowAdvice(advice, LocalContext.GetMe(runState));
                RenderLatchedPeek();

                // The screen this reply answers, for `pause_on`'s ids and the
                // mod's own verdict; combat has neither.
                SimScreenDecision? forDecision = !isCombat && _lastScreen != null &&
                                                 ReferenceEquals(_lastScreen.Screen, forScreen)
                    ? _lastScreen
                    : null;
                if (_autoPlay && !PauseForPlayer(forDecision, advice.HandBack ?? forDecision?.HandBack) &&
                    _lastBest != null)
                {
                    // Small grace so the human can see the recommendation
                    // (and press F7 to stop) before it fires.
                    ScheduleAutoPlay(isCombat ? AllyWaitMs(advice) : 0);
                }
                if (isCombat)
                {
                    _noMoveInARow = _lastBest != null ? 0 : _noMoveInARow + 1;
                    if (_autoPlay && _lastBest == null)
                    {
                        OnNoMoveInCoop(advice.Note);
                    }
                }
            });
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Bad advice reply: {ex.Message}");
        }
    }

    // ------------------------------------------------------------------
    // Phase 7: takeover ("AI takes this decision" / "AI plays until I stop")
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // F5 peek (Coach style)
    // ------------------------------------------------------------------

    /// <summary>
    /// F5 in the Coach style (SimActions.Peek gates the style): show the AI's
    /// pick for the current decision only. With advice for this position in
    /// hand it renders now; otherwise "thinking…" until the reply lands. The
    /// decision is then reported with <c>peeked: true</c>. Main thread.
    /// </summary>
    public void Peek()
    {
        if (!SimRuntime.CoachStyleActive)
        {
            return;
        }
        // _lastBest is non-null exactly while a fresh recommendation stands
        // for the current board (every path that schedules a new request
        // clears it), so it is the "advice is ready" test.
        if (_lastBest != null && TryRenderPeek())
        {
            return;
        }
        _peekLatched = true;
        SimHud.Instance.ShowPeekThinking();
        Logger.Log("[SimAdvisor] Peek: waiting for the advice for this position.");
    }

    private void RenderLatchedPeek()
    {
        if (!_peekLatched)
        {
            return;
        }
        _peekLatched = false;
        if (SimRuntime.CoachStyleActive)
        {
            TryRenderPeek();
        }
    }

    private bool TryRenderPeek()
    {
        if (!SimHud.Instance.ShowPeekNow())
        {
            return false;
        }
        // The grade of this decision carries "(peeked)".
        if (_shownRequestId != null)
        {
            _shownPeeked = true;
            if (!_shownIsCombat)
            {
                SimChoiceCapture.OnPeekShown();
            }
        }
        Logger.Log($"[SimAdvisor] Peek shown: {_lastBestLabel ?? "?"}");
        return true;
    }

    /// <summary>F6: execute the current recommendation once. Main thread
    /// (hotkeys arrive via NHotkeyManager's CallDeferred).</summary>
    public void ExecuteBestOnce()
    {
        try
        {
            if (SimRuntime.Mode != SimMode.Advisor)
            {
                return;
            }
            SimBestAction? best = _lastBest;
            if (best == null)
            {
                Logger.Log("[SimAdvisor] No fresh recommendation to execute (wait for advice).");
                return;
            }

            // Screen decisions (card reward, Armaments-style card select,
            // loot) are executed by clicking the live screen, not by queuing
            // a combat action -- and they can be open mid-combat, so they
            // have to be handled before the combat guards below.
            if (best.Action is "choose" or "skip")
            {
                ExecuteScreenChoice(best);
                return;
            }

            CombatManager cm = CombatManager.Instance;
            if (!cm.IsInProgress || cm.PlayerActionsDisabled)
            {
                return;
            }
            RunState? runState = RunManager.Instance.DebugOnlyGetState();
            Player? player = LocalContext.GetMe(runState);
            if (player == null)
            {
                return;
            }
            // Backstop for the request-side gate in RequestCombatAdviceNow:
            // advice computed for one turn must not fire into the next
            // turn's setup, where an `end_turn` is accepted (and deferred)
            // before the hand is even drawn.
            if (player.PlayerCombatState?.Phase != PlayerTurnPhase.Play)
            {
                Logger.Log($"[SimAdvisor] Takeover: not in the play phase ({player.PlayerCombatState?.Phase}); holding.");
                return;
            }

            _lastBest = null; // consume: never double-fire the same advice
            string label = string.IsNullOrEmpty(best.Label) ? best.Action : best.Label;

            switch (best.Action)
            {
                case "play":
                {
                    CardPile hand = PileType.Hand.GetPile(player);
                    if (best.CardIndex < 0 || best.CardIndex >= hand.Cards.Count)
                    {
                        Logger.Log($"[SimAdvisor] Takeover: card_index {best.CardIndex} out of range.");
                        return;
                    }
                    var card = hand.Cards[best.CardIndex];
                    // Ask the game FIRST, exactly as the autopilot lane does
                    // (SimCombatHandler.cs:213). Enqueuing a play the game will
                    // refuse produces no error and no board change -- just a
                    // `Takeover executed:` line for something that never
                    // happened. Naming the reason and the preventer here is the
                    // whole diagnosis when the search and the live game
                    // disagree about what is legal (per-turn gates that a
                    // rebuilt state reads as zero are the usual cause: see
                    // `cards_played_this_turn` in SimCombatHandler).
                    if (!card.CanPlay(out UnplayableReason reason, out AbstractModel preventer))
                    {
                        Logger.Log(
                            $"[SimAdvisor] Takeover: the game refuses {card.Id.Entry} " +
                            $"(reason={reason}, preventer={preventer?.Id.Entry ?? "none"}, " +
                            $"energy={player.PlayerCombatState?.Energy ?? -1}).");
                        OnActionRefused(player);
                        return;
                    }
                    var target = SimCombatHandler.ResolveTarget(card, best.TargetIndex);
                    if ((card.TargetType == TargetType.AnyEnemy || card.TargetType == TargetType.AnyAlly) &&
                        target == null)
                    {
                        Logger.Log("[SimAdvisor] Takeover: recommended target no longer hittable.");
                        OnActionRefused(player);
                        return;
                    }
                    // NOT reset here. Enqueuing is not acceptance: a silently
                    // refused play passes CanPlay, enqueues, and changes
                    // nothing -- so resetting on enqueue makes the counter
                    // oscillate 0 -> 1 -> 0 forever and MaxRefusedBeforeEndTurn
                    // can never be reached. Run 048 logged "(1 in a row)" 305
                    // times on one Shiv while the escape hatch sat unused.
                    // The reset lives where the board actually moved: the
                    // history check in WatchForRefusedAction.
                    _aiActionsPending++;
                    RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                        new PlayCardAction(card, target));
                    break;
                }
                case "end_turn":
                    EndTurnAsAi(player);
                    break;
                case "potion":
                {
                    var potion = SimCombatHandler.SafeGetPotionAtSlot(player, best.Slot);
                    if (potion == null)
                    {
                        Logger.Log($"[SimAdvisor] Takeover: no potion in slot {best.Slot}.");
                        return;
                    }
                    var target = SimCombatHandler.ResolvePotionTarget(player, potion, best.TargetIndex);
                    // Same guard as a card: the game drops a targeted potion
                    // with no target in silence (an AnyAlly one has none when
                    // no teammate is alive).
                    if ((potion.TargetType == TargetType.AnyEnemy || potion.TargetType == TargetType.AnyAlly) &&
                        target == null)
                    {
                        Logger.Log("[SimAdvisor] Takeover: no valid target for the recommended potion.");
                        OnActionRefused(player);
                        return;
                    }
                    RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                        new UsePotionAction(potion, target, cm.IsInProgress));
                    break;
                }
                default:
                    Logger.Log($"[SimAdvisor] Takeover: unknown best.action '{best.Action}'.");
                    return;
            }

            SimActionLog.Record(new SimActionLogEntry
            {
                Actor = "AI",
                Act = (runState?.CurrentActIndex ?? -1) + 1,
                Floor = runState?.TotalFloor ?? 0,
                DecisionType = "takeover",
                Action = label,
            });
            Logger.Log($"[SimAdvisor] Takeover executed: {label}{NoteSuffix()}");
            WatchForRefusedAction(cm, label);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] ExecuteBestOnce: {ex.Message}");
        }
    }

    /// <summary>
    /// The search recommended something the live game will not accept. Asking
    /// again is right once -- the board may simply have moved under a slow
    /// reply -- but the search is deterministic given the same state, so a
    /// genuine disagreement about legality returns the SAME illegal move
    /// forever (14 identical refused `Defend`s in a row, floor 47,
    /// 2026-08-06). Ending the turn is the one action that is always
    /// available, so it is the escape hatch: worse play than the AI intended,
    /// but the run continues and the log names the disagreement.
    /// </summary>
    private void OnActionRefused(Player player)
    {
        _refusedInARow++;
        if (_refusedInARow < MaxRefusedBeforeEndTurn)
        {
            ScheduleAdvice();
            return;
        }
        Logger.Log(
            $"[SimAdvisor] {_refusedInARow} refused recommendations in a row; ending the turn to break the loop. " +
            "This is a legality disagreement between the search and the live game -- see the reasons above.");
        SimActionLog.Note($"AI move refused {_refusedInARow}x — ending turn");
        _refusedInARow = 0;
        try
        {
            EndTurnAsAi(player);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Could not end the turn after refusals: {ex.Message}");
        }
    }

    /// <summary>
    /// The advisor answers this combat position with no move (an engine
    /// error: content it cannot model, 2026-10-23's Dowsing). Solo, autoplay
    /// just waits and the player takes over. In co-op nobody can: the enemy
    /// turn starts only once every player has ended theirs, so a silent
    /// player freezes the whole run. After a few such replies, end the turn
    /// -- the same escape hatch as <see cref="OnActionRefused"/>.
    /// </summary>
    private void OnNoMoveInCoop(string? note)
    {
        if (!SimMp.IsMultiplayer || _noMoveInARow < MaxRefusedBeforeEndTurn)
        {
            return;
        }
        Player? player = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
        if (player == null)
        {
            return;
        }
        Logger.Log(
            $"[SimAdvisor] {_noMoveInARow} combat replies without a move ({note ?? "no note"}); " +
            "ending the turn so the co-op partners are not blocked.");
        SimActionLog.Note($"AI has no move {_noMoveInARow}x — ending turn (co-op)");
        _noMoveInARow = 0;
        try
        {
            EndTurnAsAi(player);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Could not end the turn after no-move replies: {ex.Message}");
        }
    }

    /// <summary>
    /// "Executed" above means ENQUEUED, not accepted. A card the game refuses
    /// -- unplayable, or an index that addressed a different card because the
    /// board moved under a slow reply -- is dropped by the action queue in
    /// silence: no history entry, so no `History.Changed`, so nothing ever
    /// asks for advice again. Combat then wedges forever with the log's last
    /// word being a `Takeover executed:` that never happened, which is exactly
    /// how floor 47 died on 2026-08-06 (a `Defend` with no matching
    /// `Player 1 playing card` line).
    ///
    /// The screen lane already has this safety net -- its 0.5 s poll re-asks.
    /// This is the combat lane's: if the action changed nothing and nothing is
    /// in flight, ask again against the CURRENT board.
    /// </summary>
    private void WatchForRefusedAction(CombatManager cm, string label)
    {
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return;
            }
            int mark = CombatHistoryMark();
            SceneTreeTimer timer = tree.CreateTimer(RefusedActionWatchdogSeconds);
            timer.Connect(SceneTreeTimer.SignalName.Timeout, Callable.From(() =>
            {
                try
                {
                    if (!_autoPlay || SimRuntime.Mode != SimMode.Advisor)
                    {
                        return;
                    }
                    CombatManager now = CombatManager.Instance;
                    // Mid-animation or the enemy's turn: the events that fire
                    // when the board settles will drive the next request.
                    if (!now.IsInProgress || now.PlayerActionsDisabled)
                    {
                        return;
                    }
                    // Something is already on its way, or waiting to fire.
                    if (_inflightRequestId != null || _lastBest != null)
                    {
                        return;
                    }
                    // Our end turn is queued or done. It leaves no history
                    // entry of ours, and in co-op it waits on the synced
                    // queue, so an unchanged mark here is not a refusal; a
                    // re-ask would queue plays behind the end turn.
                    Player? ending = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
                    if (_aiEndTurnQueued || (ending != null && now.IsPlayerReadyToEndTurn(ending)))
                    {
                        return;
                    }
                    if (CombatHistoryMark() != mark)
                    {
                        // The board moved: this action was accepted. THIS is
                        // acceptance, and the only place the streak may reset.
                        _refusedInARow = 0;
                        return;
                    }
                    // Counted, not just retried. An unbounded re-request is
                    // only a fix when the board is expected to move on its
                    // own; when it never will -- the action queue is blocked
                    // on a pending selection, say -- it is an infinite loop
                    // that LOOKS like progress, because every pass logs
                    // another `Takeover executed`. 2,795 of them in 2h40m
                    // (2026-08-06) went unnoticed by every watchdog for
                    // exactly that reason. Route through the same escape the
                    // pre-flight refusal path uses, so N in a row ends the
                    // turn instead of trying forever.
                    Logger.Log(
                        $"[SimAdvisor] Takeover '{label}' changed nothing on the board " +
                        $"({_refusedInARow + 1} in a row); re-requesting advice.");
                    RunState? rs = RunManager.Instance.DebugOnlyGetState();
                    Player? me = LocalContext.GetMe(rs);
                    if (me == null)
                    {
                        ScheduleAdvice();
                        return;
                    }
                    OnActionRefused(me);
                }
                catch (Exception ex)
                {
                    Logger.Log($"[SimAdvisor] Refused-action watchdog: {ex.Message}");
                }
            }));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Could not arm the refused-action watchdog: {ex.Message}");
        }
    }

    /// <summary>
    /// The standing advice's note, formatted for a takeover log line.
    ///
    /// Attached to every `Takeover executed` because that line is the whole
    /// record a post-mortem gets, and without the note it cannot distinguish
    /// the head that answered: a 4000-sim search, the run-policy net, the
    /// hand-select heuristic, and `_unranked`'s explicit non-recommendation all
    /// printed identically. It is short (the advisor's own notes run to a few
    /// dozen characters) and capped, because a log this driver reads at 20 Hz
    /// has been a liability before.
    ///
    /// Both assignment sites write it unconditionally, including to null, so a
    /// note can never outlive the advice it came with.
    /// </summary>
    private string NoteSuffix()
    {
        string? note = _lastBestNote;
        if (string.IsNullOrWhiteSpace(note))
        {
            return "";
        }
        note = note.Replace('\n', ' ').Replace('\r', ' ');
        if (note.Length > 120)
        {
            note = note.Substring(0, 117) + "...";
        }
        return $"  [{note}]";
    }

    /// <summary>
    /// Execute a screen recommendation by clicking the screen it was computed
    /// for. Re-describing first guards against the screen having closed (or
    /// changed) since the advice arrived.
    /// </summary>
    private void ExecuteScreenChoice(SimBestAction best)
    {
        SimScreenDecision? screen = _lastScreen;
        _lastBest = null; // consume: never double-fire the same advice

        SimScreenDecision? current = SimScreenAdvice.Describe();
        if (current == null)
        {
            Logger.Log("[SimAdvisor] Takeover: no screen decision is open.");
            _lastScreen = null;
            return;
        }
        // `screen` is null only in the brief window between a debounce reset
        // and the next describe; the live screen is the authority either way.
        if (screen != null && !ReferenceEquals(current.Screen, screen.Screen))
        {
            Logger.Log("[SimAdvisor] Takeover: the screen changed since the advice; ignoring.");
            _lastScreen = null;
            return;
        }

        int index = best.Action == "skip" ? current.OptionCount : best.Index;
        if (index < 0)
        {
            Logger.Log($"[SimAdvisor] Takeover: recommendation has no option index ({best.Action}).");
            return;
        }
        if (best.Action != "skip")
        {
            index = ReconcileIndex(current, best, index);
            if (index < 0)
            {
                _lastScreen = null;
                ScheduleAdvice(); // re-ask for the offer that is actually up
                return;
            }
        }

        // The window measures "the same choice, going nowhere". Combat history
        // is the board's own answer to "did anything happen": a card that
        // resolves appends an entry, and a REFUSED action -- the livelock this
        // guard exists for -- appends nothing. So a window that has seen the
        // board move is not a loop, whatever the keys look like.
        //
        // Without this, in-combat hand selection trips the guard on a healthy
        // fight. The hand node's instance id is stable for the whole combat and
        // the confirm leg's signature is constant, so every `handselect:confirm`
        // produces a byte-identical key; four legitimate select-then-confirm
        // cycles look exactly like four repeats of one choice. Silent finds this
        // in a single fight (Survivor/Adrenaline/Poisoned Stab all prompt),
        // which is why runs 001-046 on Ironclad and Defect never saw it, and
        // run 047 wedged at floor 46 four times running -- at 4.0x AND at 1.0x,
        // so it is not the speed.
        //
        // Outside combat there is no history to advance, the mark stays put, and
        // the guard keeps its full strength on exactly the screens it was
        // written for (a repeated map/reward click that silently ADDS a card).
        int historyMark = CombatHistoryMark();
        if (historyMark != _choiceWindowHistoryMark)
        {
            _recentScreenChoices.Clear();
            _choiceWindowHistoryMark = historyMark;
        }

        string key = $"{current.Screen?.GetInstanceId()}|{current.SignatureOrDefault}|{index}";

        // Rapid-repeat dedupe (autoplay only). A screen needs real time to
        // retire after a press (animation + save write), and while it lingers
        // the game's own events keep re-requesting advice for it; each reply
        // re-arms ScheduleAutoPlay. At the unattended pacing (0.10 s debounce
        // + 0.15 s grace) that machine-gunned FOUR identical presses into the
        // Neow bundle screen inside ~0.6 s, and the runaway window guard
        // below -- written for ~1 s/choice cadence -- read them as a loop and
        // disarmed autoplay, wedging the run (2026-08-10 smoke run 2). One
        // press per (screen, signature, index) per cooldown restores the
        // cadence the guard was tuned for; the delayed re-ask below is the
        // retry path for a press the game genuinely dropped (same screen
        // still standing after the cooldown -> fresh advice -> re-press).
        if (_autoPlay && key == _lastChoiceKey &&
            Time.GetTicksMsec() - _lastChoiceAtMs < (ulong)(ScreenPressCooldownSeconds * 1000))
        {
            try
            {
                if (Engine.GetMainLoop() is SceneTree retryTree)
                {
                    SceneTreeTimer retry = retryTree.CreateTimer(ScreenPressCooldownSeconds + 0.5);
                    retry.Connect(SceneTreeTimer.SignalName.Timeout, Callable.From(() =>
                    {
                        if (_autoPlay && SimRuntime.Mode == SimMode.Advisor)
                        {
                            ScheduleAdvice();
                        }
                    }));
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimAdvisor] Press-dedupe retry timer: {ex.Message}");
            }
            return;
        }
        _lastChoiceKey = key;
        _lastChoiceAtMs = Time.GetTicksMsec();

        _recentScreenChoices.Enqueue(key);
        while (_recentScreenChoices.Count > ChoiceWindow)
        {
            _recentScreenChoices.Dequeue();
        }
        int seen = 0;
        foreach (string past in _recentScreenChoices)
        {
            if (past == key)
            {
                seen++;
            }
        }
        if (seen > MaxChoicesPerWindow)
        {
            // Deliberately-looping EVENTS are the one place a repeated
            // identical choice is legitimate play: SlipperyBridge's
            // HOLD_ON_LOOP page re-offers the same (signature, index) every
            // time and the policy may keep choosing it (2026-08-10 smoke
            // run 4 wedged at floor 1 exactly there — the guard disarmed
            // autoplay with nobody at the keyboard to resume it). GH #316
            // resolved the same shape in the headless driver by FORCE-
            // FINISHING the event; do the equivalent here: press an option
            // OTHER than the looping one (every event option advances or
            // finishes the event by construction) and keep autoplay armed.
            // Non-event screens keep the stop-dead behaviour — on a deck
            // screen an arbitrary alternative click would mutate the deck.
            if (_autoPlay && current.AdviceFor == NonCombatBridgeProtocol.EventState &&
                current.OptionCount > 1)
            {
                int alternative = index == 0 ? 1 : 0;
                Logger.Log(
                    $"[SimAdvisor] Takeover: event option {index} on {current.ScreenName} chosen " +
                    $"{seen}x in the last {ChoiceWindow}; forcing option {alternative} to finish " +
                    "the loop (GH #316 semantics).");
                SimActionLog.Note(
                    $"looping event — forced option {alternative} instead of option {index}");
                _recentScreenChoices.Clear();
                _lastChoiceKey = "";
                SimChoiceCapture.MarkAiChoice(current.Screen, current.SignatureOrDefault);
                if (current.Choose(alternative))
                {
                    _lastScreen = null;
                    ScheduleAdvice();
                    return;
                }
                // The alternative was refused: fall through to the stop.
            }
            Logger.Log($"[SimAdvisor] Takeover: {current.ScreenName} came round {seen} times in the " +
                       $"last {ChoiceWindow} choices; stopping instead of looping.");
            SimActionLog.Note(
                $"stuck in a loop on {current.ScreenName} — the same choice came round {seen} times; " +
                "finish this screen by hand (F7 again to resume)");
            SetAutoPlay(false);
            _recentScreenChoices.Clear();
            return;
        }

        // Whatever this decision resolves to is the AI's pick, not the human's.
        SimChoiceCapture.MarkAiChoice(current.Screen, current.SignatureOrDefault);
        if (!current.Choose(index))
        {
            Logger.Log($"[SimAdvisor] Takeover: could not pick option {index} on {current.ScreenName}.");
            return;
        }

        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        SimActionLog.Record(new SimActionLogEntry
        {
            Actor = "AI",
            Act = (runState?.CurrentActIndex ?? -1) + 1,
            Floor = runState?.TotalFloor ?? 0,
            DecisionType = current.AdviceFor,
            Action = string.IsNullOrEmpty(best.Label) ? $"option {index}" : best.Label,
        });
        Logger.Log($"[SimAdvisor] Takeover executed on {current.ScreenName}: " +
                   $"{(string.IsNullOrEmpty(best.Label) ? index.ToString() : best.Label)}" +
                   NoteSuffix());

        // The click either closes the screen or advances it (loot claims one
        // reward at a time); either way re-describe so autoplay keeps going.
        _lastScreen = null;
        ScheduleAdvice();
    }

    /// <summary>
    /// Click the option the advice NAMED, not merely the slot it sat in.
    ///
    /// The recommendation travels as an index into the option list we sent,
    /// which is only correct while that list keeps its order. On the card
    /// screens it can legitimately change under us -- the reward screen frees
    /// the previous offer's holders and adds the new ones in one call, and a
    /// read landing mid-swap sees both -- and the failure is silent: F6 picks
    /// a real card, just not the recommended one (reported twice on
    /// 2026-08-05). The label is the redundancy that catches it: both sides
    /// spell it from the same `CardModel.Title`, so a mismatch means the list
    /// moved, and the matching slot is where the recommended card went.
    /// </summary>
    private static int ReconcileIndex(SimScreenDecision current, SimBestAction best, int index)
    {
        List<string> labels = current.OptionLabels;
        if (labels.Count == 0 || string.IsNullOrEmpty(best.Label))
        {
            return index;
        }
        string want = best.Label.Trim();
        if (index >= 0 && index < labels.Count &&
            string.Equals(labels[index].Trim(), want, StringComparison.OrdinalIgnoreCase))
        {
            return index;
        }
        for (int i = 0; i < labels.Count; i++)
        {
            if (string.Equals(labels[i].Trim(), want, StringComparison.OrdinalIgnoreCase))
            {
                Logger.Log($"[SimAdvisor] Takeover: '{want}' moved {index} -> {i} since the advice; " +
                           "clicking the named option.");
                SimActionLog.Note($"advice re-anchored: '{want}' is now option {i} (was {index})");
                return i;
            }
        }
        // Nothing on screen carries that name any more: the offer itself
        // changed, so an index into it means nothing. Say so instead of
        // clicking a card nobody recommended.
        Logger.Log($"[SimAdvisor] Takeover: '{want}' is no longer on {current.ScreenName} " +
                   $"(options: {string.Join(", ", labels)}); ignoring the stale advice.");
        return -1;
    }

    /// <summary>F7: toggle continuous takeover -- each fresh advice is
    /// executed automatically until toggled off / mode leaves Advisor.</summary>
    public void ToggleAutoPlay()
    {
        SetAutoPlay(!_autoPlay);
        Logger.Log($"[SimAdvisor] Auto-play {(_autoPlay ? "ON (F7 to stop)" : "off")}");
        SimActionLog.Note($"Auto-play {(_autoPlay ? "ON (F7 to stop)" : "off")}");
        if (!_autoPlay)
        {
            return;
        }
        // Turning it back on is the player saying "try again" -- most often
        // right after clearing by hand whatever the runaway guard stopped on.
        _recentScreenChoices.Clear();
        // ...or "play it after all" on a decision `pause_on` handed back.
        _resumedKey = _pausedKey;
        if (_lastBest != null)
        {
            ScheduleAutoPlay();
        }
        else
        {
            // Turned on while sitting on a static screen: nothing else will
            // fire an event here, so ask for the advice that starts the loop.
            ScheduleAdvice();
        }
    }

    /// <summary>The one place `_autoPlay` is written: mirrors the value into
    /// <see cref="SimRuntime.AdvisorAutoplayArmed"/>, which gates the
    /// Wait/Anim speed patches for the unattended advisor-autoplay lane.
    /// Disarming immediately restores human-speed animation.</summary>
    private void SetAutoPlay(bool value)
    {
        _autoPlay = value;
        SimRuntime.AdvisorAutoplayArmed = value;
    }

    /// <summary>
    /// `pause_on` (GH #874): stop autoplay on a decision the AI cannot really
    /// make, or one the player listed, and hand it back exactly as F7-off
    /// does. Returns true when it paused. Main thread.
    /// <paramref name="screen"/> is null for combat; <paramref name="handBack"/>
    /// is the advisor's or the mod's "not really a decision" reason.
    /// </summary>
    private bool PauseForPlayer(SimScreenDecision? screen, string? handBack)
    {
        if (!_autoPlay)
        {
            return false;
        }
        IReadOnlyList<string> pauseOn = SimConfig.Instance.EffectivePauseOn;
        if (pauseOn.Count == 0)
        {
            return false;
        }
        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        string? eventId = CurrentEventId(runState);
        string adviceFor = screen?.AdviceFor ?? "combat_action";
        string? reason = AutoplayPause.Reason(pauseOn, handBack, new[] { adviceFor, screen?.ScreenName, eventId });
        if (reason == null)
        {
            return false;
        }
        string key = PauseKey(screen, adviceFor, eventId, runState);
        if (key == _resumedKey)
        {
            return false;
        }
        _pausedKey = key;
        SetAutoPlay(false);
        string toast = AutoplayPause.Toast(AutoplayPause.Pretty(eventId ?? adviceFor), reason);
        Logger.Log($"[SimAdvisor] Paused for the player on {screen?.ScreenName ?? "combat"}: {reason}");
        SimActionLog.Note(toast);
        // After ShowAdvice, which set the toolbar note to the advice's own.
        SimToolbar.Instance.SetNote(toast);
        return true;
    }

    /// <summary>
    /// One handed-back decision, however many requests it takes: an event's
    /// pages share the room node and the floor, the crystal sphere's reveals
    /// share their screen. The floor tells apart the long-lived nodes (the
    /// event room, the map) from one room to the next.
    /// </summary>
    private static string PauseKey(SimScreenDecision? screen, string adviceFor, string? eventId, RunState? runState) =>
        $"{adviceFor}|{eventId}|{runState?.TotalFloor}|" +
        (screen?.Screen is { } node && GodotObject.IsInstanceValid(node) ? node.GetInstanceId() : 0UL);

    /// <summary>The current event's id ("NEOW", "CRYSTAL_SPHERE"), or null
    /// outside an event room. Never throws.</summary>
    private static string? CurrentEventId(RunState? runState)
    {
        try
        {
            return runState?.CurrentRoom is EventRoom room ? room.LocalMutableEvent?.Id.Entry : null;
        }
        catch (Exception)
        {
            // Between pages the event synchronizer has no local event yet.
            return null;
        }
    }

    /// <summary>
    /// The co-op hold for this reply, within the round's budget (0 = none).
    /// Single-player and older advisors never set it.
    /// </summary>
    private int AllyWaitMs(SimAdvice advice)
    {
        if (advice.WaitForAllyMs <= 0 || !SimMp.IsMultiplayer)
        {
            return 0;
        }
        int round = CombatManager.Instance.DebugOnlyGetState()?.RoundNumber ?? 0;
        if (round != _allyWaitRound)
        {
            _allyWaitRound = round;
            _allyWaitUsedMs = 0;
        }
        int ms = Math.Min(advice.WaitForAllyMs, AllyWaitBudgetMs - _allyWaitUsedMs);
        if (ms <= 0)
        {
            return 0;
        }
        _allyWaitUsedMs += ms;
        Logger.Log($"[SimAdvisor] Co-op: holding {_lastBestLabel} up to {ms} ms for the teammate ({advice.Note}).");
        return ms;
    }

    private void ScheduleAutoPlay(int extraMs = 0)
    {
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return;
            }
            int gen = ++_autoPlayGen;
            // 0.6 s is the "human can read it and press F7 to veto" grace.
            // Unattended, there is no reader; 0.15 s still decouples the
            // press from the advice render (same-frame execution was never
            // exercised) without paying half a second per decision.
            SceneTreeTimer timer = tree.CreateTimer((Unattended ? 0.15 : 0.6) + extraMs / 1000.0);
            timer.Connect(SceneTreeTimer.SignalName.Timeout, Callable.From(() =>
            {
                if (gen == _autoPlayGen && _autoPlay && SimRuntime.Mode == SimMode.Advisor)
                {
                    ExecuteBestOnce();
                }
            }));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] ScheduleAutoPlay: {ex.Message}");
        }
    }

    // ------------------------------------------------------------------
    // Human actions (main thread, from History.Changed)
    // ------------------------------------------------------------------

    /// <summary>
    /// Combat history length, or -1 when there is no combat to ask. Never
    /// throws: this only feeds the loop guard, and a guard that can crash the
    /// takeover path is worse than one that occasionally reads -1.
    /// </summary>
    private static int CombatHistoryMark()
    {
        try
        {
            // Count() not .Count: Entries is an IEnumerable, so `?.Count` binds
            // the LINQ method group and does not compile.
            IEnumerable<CombatHistoryEntry>? entries = CombatManager.Instance?.History?.Entries;
            if (entries == null)
            {
                return -1;
            }
            if (!SimMp.IsMultiplayer)
            {
                return entries.Count();
            }
            // Co-op: teammates' plays append entries too, which would read as
            // "our action resolved" to every guard keyed on this mark.
            var me = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState())?.Creature;
            return entries.Count(e => e.Actor == me);
        }
        catch (Exception)
        {
            return -1;
        }
    }

    private void ReportNewHumanActions()
    {
        CombatManager cm = CombatManager.Instance;
        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        Player? player = LocalContext.GetMe(runState);
        var playerCreature = player?.Creature;

        List<CombatHistoryEntry> entries = cm.History.Entries.ToList();
        if (SimCombatUndo.Replaying)
        {
            // The undo replay re-plays recorded moves through the normal
            // queue; swallowing them here keeps them out of the wire and
            // the workbench event log (they already happened once).
            _historySeen = entries.Count;
            return;
        }
        if (entries.Count < _historySeen)
        {
            _historySeen = 0; // history was cleared (new combat)
        }

        for (int i = _historySeen; i < entries.Count; i++)
        {
            CombatHistoryEntry entry = entries[i];
            if (!ShouldLog(entry))
            {
                continue;
            }

            string detail = SafeDescription(entry);
            bool byPlayer = playerCreature != null && entry.Actor == playerCreature;
            bool isOwnAction = byPlayer && entry is CardPlayFinishedEntry or PotionUsedEntry;

            // Everything that is not a decision by us (damage, enemy moves,
            // block, powers) is world commentary -- the "what happened"
            // half the workbench log shows and this one was missing.
            if (!isOwnAction)
            {
                SimActionLog.Record(new SimActionLogEntry
                {
                    Actor = "game",
                    Act = (runState?.CurrentActIndex ?? -1) + 1,
                    Floor = runState?.TotalFloor ?? 0,
                    DecisionType = "",
                    Action = detail,
                });
                continue;
            }

            // Actions enqueued by the takeover path show up in the same
            // history feed; attribute them to the AI. Still reported on the
            // wire (the workbench event log was full of unlabeled combat
            // steps because AI plays sent nothing), just marked as the AI's.
            if (_aiActionsPending > 0)
            {
                _aiActionsPending--;
                SimActionLog.Record(new SimActionLogEntry
                {
                    Actor = "AI",
                    Act = (runState?.CurrentActIndex ?? -1) + 1,
                    Floor = runState?.TotalFloor ?? 0,
                    DecisionType = "combat",
                    Action = detail,
                });
                SendHumanAction("ai_play", $"AI: {detail}", PlayOf(entry), DrinkOf(entry));
                continue;
            }

            SimActionLog.Record(new SimActionLogEntry
            {
                Actor = "You",
                Act = (runState?.CurrentActIndex ?? -1) + 1,
                Floor = runState?.TotalFloor ?? 0,
                DecisionType = "combat",
                Action = detail,
                Advice = _lastBestLabel != null ? $"AI: {_lastBestLabel}" : null,
            });

            SendHumanAction("card_play", detail, PlayOf(entry), DrinkOf(entry));
        }
        _historySeen = entries.Count;
    }

    /// <summary>One in-combat action report on the wire (fire-and-forget) --
    /// human plays as ``card_play``, takeover plays as ``ai_play``.
    ///
    /// <para>When the action was a card play, it also carries the MACHINE
    /// form: <c>card</c> (<c>id</c> + <c>upgraded</c>) and <c>target_index</c>
    /// in <see cref="SimCombatHandler.ResolveTarget"/>'s frame (the full
    /// <c>Enemies</c> list, same order the combat payload uses). <c>detail</c>
    /// alone is a display string ("Bash -> Cultist"), which is not enough for
    /// a client to REPLAY the action -- and replaying is what lets the
    /// advisor keep an engine mirror in step with the fight. Without it the
    /// advisor could only re-seed from each new board, so it predicted
    /// nothing and every ordinary change read as a divergence: 2232 desyncs
    /// over 540 decisions on live run 001-ironclad-a8-0824-1939, 42x the
    /// autopilot's rate. Additive: a client that ignores the new fields sees
    /// exactly the old message.</para></summary>
    private void SendHumanAction(
        string action, string detail, CardPlay? play = null, PotionUsedEntry? drink = null)
    {
        (string? adviceId, long positionId, bool peeked) = TakeCombatCorrelation();
        var msg = new Dictionary<string, object?>
        {
            ["type"] = "human_action",
            ["state_id"] = _inflightStateId >= 0 ? _inflightStateId : null,
            ["decision_type"] = "combat_action",
            ["action"] = action,
            ["detail"] = detail,
            // v5: the advice rendered for exactly this position, or null.
            ["advice_request_id"] = adviceId,
            ["position_id"] = positionId,
            ["peeked"] = peeked,
            ["card"] = play?.Card == null
                ? null
                : new Dictionary<string, object?>
                {
                    ["id"] = play.Card.Id.Entry,
                    ["upgraded"] = play.Card.CurrentUpgradeLevel > 0,
                },
            // No `upgraded` twin: a potion has no upgrade state, so the id
            // alone identifies it. Two belt slots holding the same potion are
            // interchangeable, which is why no slot is sent either -- the
            // client locates it by identity in its OWN belt, the same way it
            // locates a card in its own hand, so the two sides never have to
            // agree on belt order.
            ["potion"] = drink?.Potion == null
                ? null
                : new Dictionary<string, object?> { ["id"] = drink.Potion.Id.Entry },
            ["target_index"] = drink != null ? PotionTargetIndexOf(drink) : TargetIndexOf(play),
        };
        SendHumanActionMessage(msg);
    }

    private static void SendHumanActionMessage(Dictionary<string, object?> msg)
    {
        try
        {
            BridgeServer.Instance.SendState(JsonSerializer.Serialize(msg));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Could not send human_action: {ex.Message}");
        }
    }

    // ------------------------------------------------------------------
    // v5 coach capture: correlation, end turn, non-combat choices
    // ------------------------------------------------------------------

    /// <summary>`capabilities` this mod build advertises on every
    /// advice_request (PROTOCOL.md section 5): coach 2 = chips plus the v5.1
    /// recap / autopsy records.</summary>
    private static Dictionary<string, object> ModCapabilities() => new() { ["coach"] = 2 };

    /// <summary>The run's choice synchronizer <see cref="WatchPlayerChoices"/>
    /// listens to; a new run brings a new one.</summary>
    private static PlayerChoiceSynchronizer? _choiceSync;

    /// <summary>
    /// Log every player's mid-combat choice onto the combat log
    /// (<see cref="SimCombatLog.RecordChoice"/>): the synchronizer raises
    /// `PlayerChoiceReceived` for our picks and for each teammate's as they
    /// arrive, so a recorded co-op fight can replay them. Subscribes once per
    /// run (the old run's synchronizer goes with it). Called before every
    /// drain; never throws.
    /// </summary>
    private static void WatchPlayerChoices()
    {
        try
        {
            PlayerChoiceSynchronizer? sync = RunManager.Instance?.PlayerChoiceSynchronizer;
            if (sync == null || ReferenceEquals(sync, _choiceSync))
            {
                return;
            }
            _choiceSync = sync;
            sync.PlayerChoiceReceived += (player, choiceId, result) =>
            {
                var cards = new List<Dictionary<string, object?>>();
                try
                {
                    foreach (CardModel c in result.canonicalCards ?? new())
                    {
                        cards.Add(SimCombatLog.ChoiceCard(c.Id.Entry, c.IsUpgraded));
                    }
                    foreach (var c in result.combatCards ?? new())
                    {
                        CardModel? card = c.ToCardModelOrNull();
                        cards.Add(SimCombatLog.ChoiceCard(card?.Id.Entry, card?.IsUpgraded ?? false));
                    }
                    foreach (var c in result.deckCards ?? new())
                    {
                        CardModel card = c.ToCardModel(player);
                        cards.Add(SimCombatLog.ChoiceCard(card.Id.Entry, card.IsUpgraded));
                    }
                    foreach (var c in result.mutableCards ?? new())
                    {
                        cards.Add(SimCombatLog.ChoiceCard(c.Id?.Entry, c.CurrentUpgradeLevel > 0));
                    }
                }
                catch (Exception ex)
                {
                    Logger.Log($"[SimAdvisor] Could not read a player choice's cards: {ex.Message}");
                }
                var invariant = System.Globalization.CultureInfo.InvariantCulture;
                SimCombatLog.RecordChoice(
                    player.NetId.ToString(invariant),
                    choiceId,
                    result.type.ToString(),
                    cards,
                    result.indexes?.ToList(),
                    result.playerId?.ToString(invariant));
            };
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Could not watch player choices: {ex.Message}");
        }
    }

    /// <summary>
    /// An inbound `coach` record (BridgeServer.CoachReceived, socket thread):
    /// parse here, render on the main thread. A withdrawal hides the chips it
    /// names; anything else goes through CoachChipText, which returns nothing
    /// for good turns unless the player asked for a check mark. A turn chip
    /// that arrives once the NEXT player turn is already under way is late
    /// and is dropped -- never show a late chip.
    /// </summary>
    private void OnCoachReceived(JsonElement record)
    {
        System.Text.Json.Nodes.JsonObject? rec;
        try
        {
            rec = System.Text.Json.Nodes.JsonNode.Parse(record.GetRawText()) as System.Text.Json.Nodes.JsonObject;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Bad coach record: {ex.Message}");
            return;
        }
        if (rec == null)
        {
            return;
        }
        RunOnMainThread(() =>
        {
            if (SimRuntime.Mode != SimMode.Advisor)
            {
                return;
            }
            string kind = rec["kind"]?.GetValueKind() == JsonValueKind.String ? rec["kind"]!.GetValue<string>() : "";
            if (kind == "withdraw")
            {
                List<string> ids = rec["ids"] is System.Text.Json.Nodes.JsonArray arr
                    ? arr.Where(n => n?.GetValueKind() == JsonValueKind.String).Select(n => n!.GetValue<string>()).ToList()
                    : new List<string>();
                SimCoachChip.Instance.Withdraw(ids);
                SimCoachPanel.Instance.Withdraw(ids);
                return;
            }
            if (kind is "recap" or "autopsy")
            {
                PanelText? panel = CoachPanelText.Format(rec);
                if (panel != null)
                {
                    Logger.Log($"[SimAdvisor] Coach {kind}: {panel.Title} | {panel.Summary} | {panel.Sections.Count} section(s)");
                    SimCoachPanel.Instance.Show(panel);
                }
                return;
            }
            bool inCombat = CombatManager.Instance.IsInProgress;
            if (kind == "turn" && inCombat && IsLateTurnRecord(rec))
            {
                Logger.Log("[SimAdvisor] Coach: turn record arrived after the next turn started; not shown.");
                return;
            }
            ChipText? text = CoachChipText.Format(rec, SpireMentorModConfig.CoachShowGoodTurns);
            if (text == null)
            {
                return;
            }
            Logger.Log($"[SimAdvisor] Coach chip: {text.Title} | {string.Join(" | ", text.Lines)}");
            SimCoachChip.Instance.Show(text, inCombat);
        });
    }

    /// <summary>The record grades turn N while the player is already in the
    /// play phase of a later turn.</summary>
    private static bool IsLateTurnRecord(System.Text.Json.Nodes.JsonObject rec)
    {
        try
        {
            if (rec["turn"] is not System.Text.Json.Nodes.JsonValue tv || !tv.TryGetValue(out int turn))
            {
                return false;
            }
            if (CombatManager.Instance.IsEnemyTurnStarted)
            {
                return false;
            }
            Player? me = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
            int now = me?.PlayerCombatState?.TurnNumber ?? 0;
            return now > turn;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>`capabilities.coach` of a reply (top level, next to `advice`,
    /// as the advisor sends it; `advice.capabilities` is accepted too), 0 when
    /// absent or not an integer.</summary>
    private static int CoachCapabilityOf(JsonElement root)
    {
        JsonElement caps = default;
        bool found = root.TryGetProperty("capabilities", out caps) ||
                     (root.TryGetProperty("advice", out JsonElement advice) &&
                      advice.ValueKind == JsonValueKind.Object &&
                      advice.TryGetProperty("capabilities", out caps));
        if (found &&
            caps.ValueKind == JsonValueKind.Object &&
            caps.TryGetProperty("coach", out JsonElement coach) &&
            coach.ValueKind == JsonValueKind.Number &&
            coach.TryGetInt32(out int level))
        {
            return Math.Max(0, level);
        }
        return 0;
    }

    private void ClearActedFrom()
    {
        _actedFromRequestId = null;
        _actedFromStateId = -1;
        _actedFromPeeked = false;
    }

    private void ForgetShownAdvice()
    {
        _shownRequestId = null;
        _shownStateId = -1;
        _shownPeeked = false;
        ClearActedFrom();
    }

    /// <summary>
    /// The advice a combat action is graded against: the slot of the position
    /// the player just left, else the one still on screen (an action reported
    /// before anything rescheduled), else none. Consumes the left slot, so one
    /// advice grades one action. <c>position_id</c> is that advice's state id;
    /// without advice it is a fresh id that matches no advice by construction.
    /// </summary>
    private (string? adviceId, long positionId, bool peeked) TakeCombatCorrelation()
    {
        string? id = null;
        long stateId = -1;
        bool peeked = false;
        if (_actedFromRequestId != null && _actedFromIsCombat)
        {
            id = _actedFromRequestId;
            stateId = _actedFromStateId;
            peeked = _actedFromPeeked;
        }
        else if (_shownRequestId != null && _shownIsCombat)
        {
            id = _shownRequestId;
            stateId = _shownStateId;
            peeked = _shownPeeked;
        }
        ClearActedFrom();
        if (id == null || stateId < 0)
        {
            return (null, SimProtocolHelpers.NextStateId(), false);
        }
        return (id, stateId, peeked);
    }

    /// <summary>
    /// PlayerCmd.EndTurn prefix (HumanEndTurnPatch). Only the human's end
    /// turn counts: the end-turn button's EndPlayerTurnAction is the one
    /// caller passing canBackOut: true (EndPlayerTurnAction.cs:40, both
    /// v0.107.1 and v0.111.0); card effects (Void Form), a dead player and
    /// every mod path pass false, and the mod's own calls also run under
    /// <see cref="SuppressEndTurnReport"/>. The undo replay re-enqueues
    /// recorded end turns, so it is excluded too.
    /// </summary>
    public static void OnEndTurnRequested(Player player, bool canBackOut)
    {
        try
        {
            if (SuppressEndTurnReport || !canBackOut || SimCombatUndo.Replaying ||
                SimRuntime.Mode != SimMode.Advisor || !LocalContext.IsMe(player))
            {
                return;
            }
            if (_aiEndTurnQueued)
            {
                _aiEndTurnQueued = false;
                return;
            }
            HumanEndTurnPending = true;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] OnEndTurnRequested: {ex.Message}");
        }
    }

    /// <summary>Every PlayerCmd.EndTurn the mod itself issues goes through
    /// here, so the end-turn capture never reports it as the human's.</summary>
    internal static void EndTurnAsAi(Player player)
    {
        if (SimMp.IsMultiplayer)
        {
            // PlayerCmd.EndTurn only marks this machine ready; the other
            // players never hear of it and the turn never ends. The end-turn
            // button's path (NEndTurnButton) goes through the synced queue.
            // Once: a re-ask before the action runs would queue a second one,
            // which reaches the prefix after the marker is spent.
            if (_aiEndTurnQueued || player.PlayerCombatState is not { } combatState ||
                CombatManager.Instance.IsPlayerReadyToEndTurn(player))
            {
                return;
            }
            _aiEndTurnQueued = true;
            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                new EndPlayerTurnAction(player, combatState.TurnNumber));
            return;
        }
        bool was = SuppressEndTurnReport;
        SuppressEndTurnReport = true;
        try
        {
            PlayerCmd.EndTurn(player, canBackOut: false);
        }
        finally
        {
            SuppressEndTurnReport = was;
        }
    }

    /// <summary>The human's end turn, once the enemy side has started. Sent
    /// only to an advisor that advertised coach support: an older advisor's
    /// mirror DROPS on any card-less combat action.</summary>
    private void SendEndTurn()
    {
        if (AdvisorCoachCapability < 1 || SimCombatUndo.Replaying)
        {
            ClearActedFrom();
            return;
        }
        (string? adviceId, long positionId, bool peeked) = TakeCombatCorrelation();
        SendHumanActionMessage(new Dictionary<string, object?>
        {
            ["type"] = "human_action",
            ["state_id"] = null,
            ["decision_type"] = "combat_action",
            ["action"] = "end_turn",
            ["detail"] = "end turn",
            ["advice_request_id"] = adviceId,
            ["position_id"] = positionId,
            ["peeked"] = peeked,
            ["card"] = null,
            ["potion"] = null,
            ["target_index"] = null,
        });
    }

    /// <summary>
    /// One non-combat pick (SimChoiceCapture): <paramref name="index"/> in the
    /// advice's option index space, null for a skip. Gated like the end turn.
    /// </summary>
    internal void SendChoice(
        string decisionType, int? index, string label, string? adviceRequestId, long adviceStateId, bool peeked)
    {
        if (SimRuntime.Mode != SimMode.Advisor || AdvisorCoachCapability < 1 || SimCombatUndo.Replaying)
        {
            return;
        }
        bool hasAdvice = adviceRequestId != null && adviceStateId >= 0;
        Logger.Log($"[SimAdvisor] Human pick on {decisionType}: {(index?.ToString() ?? "skip")} '{label}'" +
                   $" (advice {adviceRequestId ?? "none"})");
        SendHumanActionMessage(new Dictionary<string, object?>
        {
            ["type"] = "human_action",
            ["state_id"] = null,
            ["decision_type"] = decisionType,
            ["action"] = index == null ? "skip" : "choose",
            ["detail"] = label,
            ["advice_request_id"] = hasAdvice ? adviceRequestId : null,
            ["position_id"] = hasAdvice ? adviceStateId : SimProtocolHelpers.NextStateId(),
            ["peeked"] = hasAdvice && peeked,
            ["choice"] = new Dictionary<string, object?> { ["index"] = index, ["label"] = label },
        });
    }

    /// <summary>The `CardPlay` behind a history entry, when it was a card
    /// play. A potion use returns null here and carries its own machine form
    /// through <see cref="DrinkOf"/> instead.</summary>
    private static CardPlay? PlayOf(CombatHistoryEntry entry) =>
        entry is CardPlayFinishedEntry finished ? finished.CardPlay : null;

    /// <summary>The `PotionUsedEntry` behind a history entry, when it was a
    /// drink.
    ///
    /// <para>This used to return nothing at all: a potion use was reported
    /// with `card` null, which the advisor correctly read as "a real board
    /// change I cannot replay" and answered by dropping its mirror. Honest,
    /// but expensive -- every fight in which anything was drunk lost its
    /// mirror at that moment, and with it every searched decision for the rest
    /// of the fight. The engine models the drink as a first-class action
    /// (`Action::DrinkPotion`), so the id is all a client needs to follow
    /// it.</para></summary>
    private static PotionUsedEntry? DrinkOf(CombatHistoryEntry entry) =>
        entry as PotionUsedEntry;

    /// <summary>The played card's target as an index into the full
    /// <c>Enemies</c> list, or null when the card took no target.
    ///
    /// <para>Deliberately the SAME frame <see cref="SimCombatHandler.
    /// ResolveTarget"/> reads, not the hittable-only list: a corpse keeps its
    /// slot there, so the index stays stable across a kill within the same
    /// board. Null on any lookup failure, which the client must treat as "not
    /// replayable" rather than "untargeted".</para></summary>
    private static int? TargetIndexOf(CardPlay? play)
    {
        try
        {
            Creature? target = play?.Target;
            if (target == null)
            {
                return null;
            }
            List<Creature>? enemies = play?.Card?.CombatState?.Enemies?.ToList();
            int idx = enemies?.IndexOf(target) ?? -1;
            return idx >= 0 ? idx : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>The drink's target as an index into the full <c>Enemies</c>
    /// list, in the same frame <see cref="TargetIndexOf"/> uses, or null when
    /// the potion took no target.
    ///
    /// <para>Advisory only: the engine resolves a null potion target to the
    /// first alive enemy exactly as C# `RunSimulator.DoUsePotion` does, and no
    /// potion that opens a card-select is single-enemy targeted. Sent anyway
    /// so a targeted drink (FirePotion) is followed against the enemy it
    /// actually hit rather than the default.</para></summary>
    private static int? PotionTargetIndexOf(PotionUsedEntry drink)
    {
        try
        {
            Creature? target = drink.Target;
            if (target == null)
            {
                return null;
            }
            List<Creature>? enemies =
                drink.Potion?.Owner?.Creature?.CombatState?.Enemies?.ToList();
            int idx = enemies?.IndexOf(target) ?? -1;
            return idx >= 0 ? idx : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// Which history entries reach the action log. Every entry already
    /// renders itself (`CombatHistoryEntry.Description`), so this is purely a
    /// noise filter: card draw / discard / energy spend fire several times
    /// per turn and say nothing about what happened.
    /// </summary>
    private static bool ShouldLog(CombatHistoryEntry entry) => entry switch
    {
        CardDrawnEntry => false,
        CardDiscardedEntry => false,
        EnergySpentEntry => false,
        CardPlayStartedEntry => false, // the Finished twin carries the outcome
        _ => true,
    };

    private static string SafeDescription(CombatHistoryEntry entry)
    {
        try
        {
            return entry.Description;
        }
        catch
        {
            return entry.GetType().Name;
        }
    }

    // ------------------------------------------------------------------

    private static void RunOnMainThread(Action action)
    {
        try
        {
            if (NGame.IsMainThread())
            {
                Guarded(action);
            }
            else
            {
                Callable.From(() => Guarded(action)).CallDeferred();
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Dispatch failed: {ex.Message}");
        }
    }

    private static void Guarded(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisor] Ignoring error (never breaks gameplay): {ex.Message}");
        }
    }
}
