// SimRestore.cs -- in-process restore of a SimSnapshots snapshot (rewind).
//
// The restore primitive is the game's own debug hot-swap
// (FileDropHandler.OnRunHistoryDropped, modded-gated, v0.107.1):
//
//     RunManager.Instance.CleanUp();
//     RunState state = RunState.FromSerializable(save);
//     await RunManager.Instance.SetUpSavedSingleplayer(state, save);
//     NGame.Instance.ReactionContainer.InitializeNetworking(new NetSingleplayerGameService());
//     await NGame.Instance.LoadRun(state, save.PreFinishedRoom);
//
// Sub-second, no file I/O, no main-menu round trip. RNG counters restore
// bit-exact (RunRngSet.FromSave fast-forwards fresh streams).
//
// Two deliberate side effects we accept:
//   * SetUpSavedSingleplayer bumps NumReloads and re-saves current_run.save
//     with the restored state -- that IS the semantic ("the run continues
//     from here"), and it goes through the game's own cloud-safe save path.
//   * A different line after the rewind consumes different RNG draws, so
//     downstream offers legitimately diverge (the seed fixes content baked
//     into the act maps, not the future stream order).
//
// Discipline (SimAutoSlayer's hard-won rule): CleanUp + LoadRun destroys and
// rebuilds the run scene, so NOTHING here holds a game node reference across
// an await, and the whole sequence runs on the main thread. After the load,
// the advisor's in-flight advice and armed best-action refer to a board that
// no longer exists -- SimAdvisor.NotifyRunRestored() invalidates them and
// schedules fresh advice.

using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace SpireMentor;

internal static class SimRestore
{
    private static int _inProgress;

    /// <summary>True while a restore is running. SimSnapshots skips its
    /// capture postfix during this window (SetUpSavedSingleplayer re-saves
    /// the run, which would otherwise record a duplicate node).</summary>
    public static bool InProgress => Volatile.Read(ref _inProgress) != 0;

    /// <summary>Claim the rewind lane. Shared with SimCombatUndo so a room
    /// rewind and a turn undo exclude each other.</summary>
    internal static bool TryBegin() => Interlocked.CompareExchange(ref _inProgress, 1, 0) == 0;

    internal static void End() => Volatile.Write(ref _inProgress, 0);

    /// <summary>
    /// The bare reload primitive (see the file header): tear the run down
    /// and load <paramref name="save"/> in place. Caller must hold the
    /// rewind lane (TryBegin) and runs on the main thread.
    /// </summary>
    internal static async Task ReloadRunFromSave(SerializableRun save)
    {
        if (SimMp.IsMultiplayer)
        {
            throw new InvalidOperationException("a co-op run cannot be reloaded as single-player");
        }
        if (RunManager.Instance.IsInProgress)
        {
            RunManager.Instance.CleanUp();
        }
        RunState state = RunState.FromSerializable(save);
        await RunManager.Instance.SetUpSavedSingleplayer(state, save);
        NGame.Instance.ReactionContainer.InitializeNetworking(new NetSingleplayerGameService());
        await NGame.Instance.LoadRun(state, save.PreFinishedRoom);
    }

    /// <summary>
    /// Rewind the live run to <paramref name="snapshotId"/>. Safe from any
    /// thread; the work runs deferred on the main thread. Outcome (success
    /// or structured error) is broadcast to the bridge client and noted in
    /// the action log.
    /// </summary>
    public static void RequestRestore(string snapshotId)
    {
        Callable.From(() => TaskHelper.RunSafely(RestoreAsync(snapshotId))).CallDeferred();
    }

    private static async Task RestoreAsync(string snapshotId)
    {
        if (SimMp.IsMultiplayer)
        {
            Fail(snapshotId, "restore_multiplayer", SimMp.RefusedMessage("Rewind"));
            return;
        }
        if (!TryBegin())
        {
            Fail(snapshotId, "restore_busy", "another restore is already running");
            return;
        }
        try
        {
            if (!RunManager.Instance.IsInProgress && NGame.Instance == null)
            {
                Fail(snapshotId, "no_active_run", "the game is not in a state that can load a run");
                return;
            }

            string? json = SimSnapshots.GetJson(snapshotId);
            if (json == null)
            {
                Fail(snapshotId, "snapshot_not_found", $"no snapshot with id {snapshotId}");
                return;
            }

            ReadSaveResult<SerializableRun> parsed = JsonSerializationUtility.FromJson<SerializableRun>(json);
            if (!parsed.Success)
            {
                Fail(snapshotId, "snapshot_corrupt", $"{parsed.ErrorMessage} ({parsed.Status})");
                return;
            }
            SerializableRun save = parsed.SaveData;

            Logger.Log($"[SimRestore] Rewinding to snapshot {snapshotId}...");
            SimActionLog.Note($"Rewinding to snapshot {snapshotId}...");

            await ReloadRunFromSave(save);

            int floor = RunManager.Instance.IsInProgress
                ? RunManager.Instance.DebugOnlyGetState().TotalFloor
                : 0;
            Logger.Log($"[SimRestore] Restored snapshot {snapshotId} (floor {floor}).");
            SimActionLog.Note($"Rewound to floor {floor}.");

            SimAdvisor.Instance.NotifyRunRestored();

            BridgeServer.Instance.SendState(JsonSerializer.Serialize(new Dictionary<string, object?>
            {
                ["type"] = "snapshot_restored",
                ["snapshot_id"] = snapshotId,
                ["floor"] = floor,
                ["state_id"] = SimProtocolHelpers.NextStateId(),
            }));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimRestore] Restore failed: {ex}");
            Fail(snapshotId, "restore_failed", ex.Message);
        }
        finally
        {
            End();
        }
    }

    private static void Fail(string snapshotId, string code, string detail)
    {
        Logger.Log($"[SimRestore] {code}: {detail}");
        SimActionLog.Note($"Rewind failed ({code}): {detail}");
        BridgeServer.Instance.SendState(JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["type"] = "error",
            ["error"] = code,
            ["detail"] = detail,
            ["snapshot_id"] = snapshotId,
        }));
    }
}
