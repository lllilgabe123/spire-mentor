// SimCombatUndo.cs -- mid-combat undo: rewind to the start of the current
// player turn (B3 spike).
//
// The game records a CombatReplay of EVERY room in normal singleplayer play
// (RunManager.cs:830/1043 -- RecordInitialState(ToSave(null)) at each room
// entry, writer enabled unless TestMode). That tape is the whole trick:
//
//   1. Grab the live tape from CombatReplayWriter (private `_replay`).
//   2. Cut it at the start of the current player turn (turn boundaries are
//      NetReadyToBeginEnemyTurnAction events; the boundary itself is the
//      FIRST player GameAction after the last one, so the enemy turn's own
//      hook/choice traffic stays inside the replayed prefix).
//   3. Reload the tape's own SerializableRun (the SimRestore primitive --
//      bit-exact room entry, same RNG counters).
//   4. Fast-forward the id counters the way the game's own debug replay
//      does (NMultiplayerTest.RunReplay), then inject the prefix.
//
// Injection differs from RunReplay in one deliberate way: RunReplay runs
// under a Replay-type net service where nothing self-generates, so it must
// inject Hook / Resume / PlayerChoice events from the tape. We stay in LIVE
// singleplayer (the run must continue normally afterwards), where
// ActionQueueSynchronizer self-enqueues hooks (NetGameType.Singleplayer ->
// EnqueueAction immediately) and choices resolve locally -- so Hook and
// Resume tape events are SKIPPED (the machinery regenerates them, with
// matching ids thanks to the fast-forward), and PlayerChoice events are
// answered through CardSelectCmd.PushSelector, the game's own automated-
// selection hook (used by TestMode, AutoSlay and WhisperingEarring), which
// bypasses the selection UI entirely.
//
// Known spike limits (documented, not fixed):
//   * Choice events that do not flow through CardSelectCmd (RelicSelectCmd
//     mid-combat -- rare) would pop their UI during the replay; the recorded
//     answer queue then stays unconsumed and the replay may need a human
//     click to finish.
//   * The replayed prefix re-runs through every observer (mirror, HUD), so
//     the workbench event log sees the combat states again; SimAdvisor's
//     human-action reporter is suppressed for the duration so replayed moves
//     are not re-reported as fresh plays.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace SpireMentor;

internal static class SimCombatUndo
{
    private static int _replaying;

    /// <summary>True while the undo replay is injecting the recorded prefix.
    /// SimAdvisor's history reporter swallows entries during this window.</summary>
    public static bool Replaying => Volatile.Read(ref _replaying) != 0;

    private static readonly FieldInfo? ReplayField =
        typeof(CombatReplayWriter).GetField("_replay", BindingFlags.NonPublic | BindingFlags.Instance);

    /// <summary>The writer's live tape for the CURRENT room, or null.</summary>
    private static CombatReplay? LiveTape()
    {
        try
        {
            return ReplayField?.GetValue(RunManager.Instance.CombatReplayWriter) as CombatReplay;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombatUndo] tape reflection failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>Undo to the start of the current player turn (or, when
    /// nothing was played yet this turn, the previous turn). Safe from any
    /// thread; runs deferred on the main thread.
    ///
    /// <paramref name="targetTurn"/> &gt;= 1 undoes to the start of that turn
    /// instead of the current one. <paramref name="tapeId"/> replays a
    /// PERSISTED tape (a past combat's -- see SimSnapshots.TapeFile) rather
    /// than the live writer's: the tape's own room-entry save is reloaded
    /// first, so this lands mid-fight in a room the run already left.</summary>
    public static void RequestUndoTurn(int targetTurn = 0, string? tapeId = null)
    {
        Callable.From(() => TaskHelper.RunSafely(UndoTurnAsync(targetTurn, tapeId))).CallDeferred();
    }

    private static async Task UndoTurnAsync(int targetTurn, string? tapeId)
    {
        if (SimMp.IsMultiplayer)
        {
            // The reload is single-player only: it would drop this peer out of
            // the session and roll back state the other players already have.
            Fail("undo_multiplayer", SimMp.RefusedMessage("Undo"));
            return;
        }
        if (!SimRestore.TryBegin())
        {
            Fail("undo_busy", "another rewind is already running");
            return;
        }
        double speedBefore = SimSpeed.Desired;
        IDisposable? selectorScope = null;
        try
        {
            CombatManager cm = CombatManager.Instance;
            RunState? runState = RunManager.Instance.DebugOnlyGetState();
            CombatReplay? tape;
            if (tapeId == null)
            {
                // Live-tape undo: only valid mid-combat on the player's turn
                // (the tape must end at a consistent, cuttable point).
                if (runState == null || !cm.IsInProgress || cm.IsEnding)
                {
                    Fail("undo_not_in_combat", "no combat in progress");
                    return;
                }
                if (cm.EndingPlayerTurnPhaseOne || cm.EndingPlayerTurnPhaseTwo ||
                    cm.DebugOnlyGetState().CurrentSide != CombatSide.Player)
                {
                    Fail("undo_not_player_turn", "wait for your turn before undoing");
                    return;
                }
                tape = LiveTape();
            }
            else
            {
                // Persisted-tape replay: a COMPLETE past combat, consistent by
                // construction, so the mid-combat guards do not apply -- the
                // reload below tears down whatever room we are in now.
                if (runState == null)
                {
                    Fail("undo_failed", "no run in progress");
                    return;
                }
                tape = LoadTape(tapeId);
            }
            if (tape?.serializableRun == null)
            {
                Fail("undo_no_tape", tapeId == null
                    ? "the game has no combat tape for this room"
                    : $"no persisted combat tape for {tapeId}");
                return;
            }

            // Copy everything we need BEFORE CleanUp (it disposes the writer).
            var events = new List<CombatReplayEvent>(tape.events);
            uint nextActionId = tape.nextActionId;
            uint nextHookId = tape.nextHookId;
            var choiceIds = new List<uint>(tape.choiceIds);
            var rewardIds = new List<int>(tape.rewardIds);
            SerializableRun save = tape.serializableRun;

            int cut = targetTurn >= 1 ? CutIndexForTurn(events, targetTurn) : CutIndex(events);
            if (cut < 0)
            {
                Fail("undo_bad_turn", $"turn {targetTurn} never happened on this combat's tape");
                return;
            }
            List<CombatReplayEvent> prefix = events.GetRange(0, cut);
            int turn = 1 + prefix.Count(e =>
                e.eventType == CombatReplayEventType.GameAction && e.action is NetEndPlayerTurnAction);
            Logger.Log($"[SimCombatUndo] Undo to turn {turn}: replaying {cut}/{events.Count} tape events.");
            SimActionLog.Note($"Undoing to the start of turn {turn}...");

            Volatile.Write(ref _replaying, 1);
            await SimRestore.ReloadRunFromSave(save);

            RunState? restored = RunManager.Instance.DebugOnlyGetState();
            if (restored == null)
            {
                Fail("undo_failed", "run did not reload");
                return;
            }

            // The debug replay's counter fast-forward (NMultiplayerTest.
            // RunReplay): recorded Resume/Hook ids only match if the counters
            // resume from the tape's room-entry baselines.
            RunManager.Instance.ActionQueueSet.FastForwardNextActionId(nextActionId);
            RunManager.Instance.ActionQueueSynchronizer.FastForwardHookId(nextHookId);
            RunManager.Instance.PlayerChoiceSynchronizer.FastForwardChoiceIds(choiceIds);
            RunManager.Instance.RewardsSetSynchronizer.FastForwardRewardIds(rewardIds);

            // The reload's fresh writer snapshotted PRE-fast-forward counter
            // baselines; patch its tape so a second undo of this combat stays
            // consistent. (CombatReplay's fields are public; only the
            // writer's `_replay` reference is not.)
            CombatReplay? fresh = LiveTape();
            if (fresh != null)
            {
                fresh.nextActionId = nextActionId;
                fresh.nextHookId = nextHookId;
                fresh.choiceIds = new List<uint>(choiceIds);
                fresh.rewardIds = new List<int>(rewardIds);
            }

            while (RunManager.Instance.ActionExecutor.IsPaused)
            {
                await NextFrame();
            }

            if (prefix.Count > 0)
            {
                var choiceQueue = new Queue<CombatReplayEvent>(prefix.Where(e =>
                    e.eventType == CombatReplayEventType.PlayerChoice));
                selectorScope = SimGameApi.PushSelector(new TapeCardSelector(choiceQueue, fresh));
                SimSpeed.Apply(SimSpeed.Max, showVfx: false);
                await InjectPrefix(prefix);
                await RunManager.Instance.ActionExecutor.FinishedExecutingActions();
            }

            int floor = restored.TotalFloor;
            Logger.Log($"[SimCombatUndo] Undo complete: floor {floor}, start of turn {turn}.");
            SimActionLog.Note($"Rewound to the start of turn {turn}.");
            SimAdvisor.Instance.NotifyRunRestored();
            BridgeServer.Instance.SendState(JsonSerializer.Serialize(new Dictionary<string, object?>
            {
                ["type"] = "combat_undone",
                ["floor"] = floor,
                ["turn"] = turn,
                ["tape_id"] = tapeId,
                ["replayed_events"] = prefix.Count,
                ["state_id"] = SimProtocolHelpers.NextStateId(),
            }));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombatUndo] Undo failed: {ex}");
            Fail("undo_failed", ex.Message);
        }
        finally
        {
            selectorScope?.Dispose();
            SimSpeed.Apply(speedBefore);
            Volatile.Write(ref _replaying, 0);
            SimRestore.End();
        }
    }

    /// <summary>Binary-serialize the live tape in the game's own replay wire
    /// format (PacketWriter -- what `replays/latest.mcr` uses), WITHOUT the
    /// anonymization pass: the undo path reloads the tape's SerializableRun
    /// as the live run, so it must stay exact. Null when there is no tape or
    /// nothing has happened yet.</summary>
    internal static byte[]? SerializeLiveTape()
    {
        try
        {
            CombatReplay? tape = LiveTape();
            if (tape?.serializableRun == null || tape.events.Count == 0)
            {
                return null;
            }
            var writer = new PacketWriter { WarnOnGrow = false };
            writer.Write(tape);
            return writer.Buffer.AsSpan(0, writer.BytePosition).ToArray();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombatUndo] tape serialize failed: {ex.Message}");
            return null;
        }
    }

    private static CombatReplay? LoadTape(string tapeId)
    {
        try
        {
            string? path = SimSnapshots.TapeFile(tapeId);
            if (path == null)
            {
                return null;
            }
            var reader = new PacketReader();
            reader.Reset(File.ReadAllBytes(path));
            return reader.Read<CombatReplay>();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombatUndo] tape load failed for {tapeId}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Index of the first tape event of turn <paramref name="turn"/> (1-based;
    /// turn 1 = the combat's first event). -1 when the tape never reached that
    /// turn -- the caller reports `undo_bad_turn` instead of guessing.
    /// </summary>
    private static int CutIndexForTurn(List<CombatReplayEvent> events, int turn)
    {
        if (turn <= 1)
        {
            return 0;
        }
        int readySeen = 0;
        for (int i = 0; i < events.Count; i++)
        {
            if (events[i].eventType == CombatReplayEventType.GameAction &&
                events[i].action is NetReadyToBeginEnemyTurnAction)
            {
                readySeen++;
                if (readySeen == turn - 1)
                {
                    for (int j = i + 1; j < events.Count; j++)
                    {
                        if (events[j].eventType == CombatReplayEventType.GameAction)
                        {
                            return j;
                        }
                    }
                    return events.Count;
                }
            }
        }
        return -1;
    }

    /// <summary>
    /// Index of the first tape event of the current player turn. Boundary =
    /// the first GameAction after the LAST NetReadyToBeginEnemyTurnAction,
    /// so the preceding enemy turn's hook/choice traffic stays in the
    /// replayed prefix. When the current turn has no actions yet (undo
    /// pressed at turn start), fall back one full turn; turn 1 falls back
    /// to 0 (= restart the combat).
    /// </summary>
    private static int CutIndex(List<CombatReplayEvent> events)
    {
        var readyIdx = new List<int>();
        for (int i = 0; i < events.Count; i++)
        {
            if (events[i].eventType == CombatReplayEventType.GameAction &&
                events[i].action is NetReadyToBeginEnemyTurnAction)
            {
                readyIdx.Add(i);
            }
        }
        int FirstActionAfter(int pos)
        {
            for (int j = pos + 1; j < events.Count; j++)
            {
                if (events[j].eventType == CombatReplayEventType.GameAction)
                {
                    return j;
                }
            }
            return events.Count;
        }
        if (readyIdx.Count == 0)
        {
            return 0;
        }
        int cut = FirstActionAfter(readyIdx[^1]);
        if (cut >= events.Count)
        {
            cut = readyIdx.Count >= 2 ? FirstActionAfter(readyIdx[^2]) : 0;
            if (cut >= events.Count)
            {
                cut = 0;
            }
        }
        return cut;
    }

    /// <summary>
    /// NMultiplayerTest.RunReplay's injection loop, minus the events that
    /// live singleplayer self-generates (Hook / Resume) and minus
    /// PlayerChoice (answered by the pushed selector when the game asks).
    /// </summary>
    private static async Task InjectPrefix(List<CombatReplayEvent> prefix)
    {
        RunState runState = RunManager.Instance.DebugOnlyGetState()!;
        foreach (CombatReplayEvent ev in prefix)
        {
            if (ev.eventType != CombatReplayEventType.GameAction || ev.action == null)
            {
                continue;
            }
            while (CombatManager.Instance.EndingPlayerTurnPhaseOne ||
                   CombatManager.Instance.EndingPlayerTurnPhaseTwo)
            {
                await NextFrame();
            }
            Player player = runState.GetPlayer(ev.playerId!.Value);
            GameAction action = ev.action.ToGameAction(player);
            if (action.ActionType == GameActionType.CombatPlayPhaseOnly)
            {
                while (CombatManager.Instance.DebugOnlyGetState().CurrentSide == CombatSide.Enemy)
                {
                    await NextFrame();
                }
            }
            RunManager.Instance.ActionQueueSet.EnqueueWithoutSynchronizing(action);
            if (action is EndPlayerTurnAction || action is ReadyToBeginEnemyTurnAction)
            {
                await RunManager.Instance.ActionExecutor.FinishedExecutingActions();
            }
        }
    }

    private static async Task NextFrame()
    {
        if (NGame.Instance is Node node && Engine.GetMainLoop() is SceneTree tree)
        {
            await node.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
        }
        else
        {
            await Task.Delay(16);
        }
    }

    private static void Fail(string code, string detail)
    {
        Logger.Log($"[SimCombatUndo] {code}: {detail}");
        SimActionLog.Note($"Undo failed ({code}): {detail}");
        try
        {
            BridgeServer.Instance.SendState(JsonSerializer.Serialize(new Dictionary<string, object?>
            {
                ["type"] = "error",
                ["error"] = code,
                ["detail"] = detail,
            }));
        }
        catch
        {
            // bridge may be disconnected; the note is enough
        }
    }

    /// <summary>
    /// Answers card selections during the replay from the recorded
    /// PlayerChoice queue, and re-records each consumed choice into the new
    /// live tape (the selector path bypasses SyncLocalChoice, so nothing
    /// would be recorded otherwise and a second undo would lose the answer).
    /// </summary>
    private sealed class TapeCardSelector : ICardSelector
    {
        private readonly Queue<CombatReplayEvent> _choices;
        private readonly CombatReplay? _recordInto;

        public TapeCardSelector(Queue<CombatReplayEvent> choices, CombatReplay? recordInto)
        {
            _choices = choices;
            _recordInto = recordInto;
        }

        public Task<IEnumerable<CardModel>> GetSelectedCards(
            IEnumerable<CardModel> options, int minSelect, int maxSelect)
        {
            List<CardModel> pool = options.ToList();
            if (_choices.Count == 0)
            {
                // Past the tape (should not happen -- the cut sits before any
                // unplayed action). Deterministic fallback beats a wedge.
                Logger.Log("[SimCombatUndo] choice queue empty; picking first option(s)");
                return Task.FromResult<IEnumerable<CardModel>>(pool.Take(Math.Max(minSelect, 0)).ToList());
            }
            CombatReplayEvent ev = _choices.Dequeue();
            _recordInto?.events.Add(ev);
            List<CardModel> picked = Resolve(ev, pool);
            Logger.Log($"[SimCombatUndo] replayed choice {ev.choiceId}: " +
                       string.Join(",", picked.Select(c => c.Id.Entry)));
            return Task.FromResult<IEnumerable<CardModel>>(picked);
        }

        public CardRewardSelection GetSelectedCardReward(
            IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
        {
            // Card rewards do not occur inside the replayed combat prefix.
            Logger.Log("[SimCombatUndo] unexpected card-reward selection during replay; skipping");
            return default;
        }

        private static List<CardModel> Resolve(CombatReplayEvent ev, List<CardModel> pool)
        {
            try
            {
                RunState runState = RunManager.Instance.DebugOnlyGetState()!;
                Player sender = runState.GetPlayer(ev.playerId!.Value);
                PlayerChoiceResult result =
                    PlayerChoiceResult.FromNetData(sender, runState, ev.playerChoiceResult!.Value);
                // Index-based forms first (most in-combat selects); card-based
                // forms resolve to models we match into the offered pool by
                // reference, then by id.
                try
                {
                    return result.AsIndexes()
                        .Where(i => i >= 0 && i < pool.Count)
                        .Select(i => pool[i])
                        .ToList();
                }
                catch (Exception)
                {
                    List<CardModel> cards =
                        (result.AsMutableCards() ?? Enumerable.Empty<CardModel>()).ToList();
                    return cards
                        .Select(c => pool.FirstOrDefault(p => ReferenceEquals(p, c))
                                     ?? pool.FirstOrDefault(p => p.Id.Equals(c.Id)))
                        .Where(c => c != null)
                        .Select(c => c!)
                        .ToList();
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimCombatUndo] choice resolve failed ({ex.Message}); picking nothing");
                return new List<CardModel>();
            }
        }
    }
}
