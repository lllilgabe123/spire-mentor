// SimSnapshots.cs -- run snapshots at the game's own save points.
//
// The rewind feature's capture side. The game writes its run save at
// exactly two in-run moments (RunManager.EnterMapPointInternal = standing
// on a map node, room not yet rolled; CombatManager.EndCombatInternal =
// post-combat with the pending reward set), plus ancient-event completion.
// Those are the ONLY moments a SerializableRun is guaranteed valid --
// ToSave() mid-room writes live-combat HP into the run save and duplicates
// MapPointHistory on reload -- so we do not snapshot on a timer; we
// postfix RunSaveManager.SaveRun(SerializableRun, bool) and keep the very
// object the game just persisted.
//
// Storage: %LOCALAPPDATA%\SpireMentor\snapshots\<start_time>\<seq>_f<floor>.run.json
// plus an index.json sidecar. Deliberately OUTSIDE the game's save tree:
// Steam cloud sync never touches it, and SaveManager.CleanupStaleCurrentRunSaves
// (which deletes current_run.save once a history/{start_time}.run exists)
// cannot reap it -- so a post-death rewind survives a game restart.
//
// The in-memory ring keeps the last MaxInMemory snapshots' JSON for
// sub-second restores; older ones are re-read from disk on demand.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Managers;

namespace SpireMentor;

internal sealed class SimSnapshotMeta
{
    public string Id { get; set; } = "";
    public int Seq { get; set; }
    public string Ts { get; set; } = "";
    public int Floor { get; set; }
    public int Act { get; set; }
    public long StartTime { get; set; }
    /// <summary>True = the post-combat save (rewards pending); false = the
    /// room-entry save (standing on the node, room not yet rolled).</summary>
    public bool PostCombat { get; set; }
    public int NumReloads { get; set; }
    /// <summary>True when the completed combat's replay tape was persisted
    /// alongside this (post-combat) snapshot -- the fight's mid-combat turns
    /// can be replayed with `undo_turn {tape_id}`.</summary>
    public bool HasTape { get; set; }
}

internal static class SimSnapshots
{
    private const int MaxInMemory = 64;

    private static readonly object Lock = new();
    private static readonly List<SimSnapshotMeta> Metas = new();
    private static readonly Dictionary<string, string> JsonById = new();
    private static long _currentStartTime = -1;
    private static int _nextSeq;

    public static string SnapshotsRoot =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "SpireMentor", "snapshots");

    /// <summary>Applied from MainFile.Initialize (unconditional -- capture is
    /// passive and costs one JSON serialize per game save).</summary>
    public static void Install(Harmony harmony)
    {
        var target = AccessTools.Method(
            typeof(RunSaveManager), nameof(RunSaveManager.SaveRun),
            new[] { typeof(SerializableRun), typeof(bool) });
        var postfix = new HarmonyMethod(typeof(SimSnapshots), nameof(OnSaveRun));
        harmony.Patch(target, postfix: postfix);
        Logger.Log("[SimSnapshots] Capturing run snapshots at the game's save points.");
    }

    /// <summary>Harmony postfix on RunSaveManager.SaveRun(SerializableRun, bool).
    /// Runs synchronously at call time (before the async write completes),
    /// which is exactly when <paramref name="save"/> is complete and unshared.</summary>
    private static void OnSaveRun(SerializableRun save, bool isMultiplayer)
    {
        try
        {
            if (isMultiplayer)
            {
                // No rewinds in co-op; forget the previous single-player
                // run's snapshots so nothing offers them as this run's.
                lock (Lock)
                {
                    Metas.Clear();
                    JsonById.Clear();
                    _currentStartTime = -1;
                }
                return;
            }
            if (save == null || SimRestore.InProgress)
            {
                return;
            }
            Capture(save);
        }
        catch (Exception ex)
        {
            // Never let a snapshot failure break the game's own save.
            Logger.Log($"[SimSnapshots] Capture failed: {ex.Message}");
        }
    }

    private static void Capture(SerializableRun save)
    {
        string json = JsonSerializationUtility.ToJson(save);
        // The completed combat's tape rides with the post-combat save: it is
        // what makes PAST mid-combat tree nodes replayable (undo_turn with
        // tape_id). Serialized synchronously -- the live writer keeps
        // appending to the same tape object, so a deferred serialize would
        // race it; the disk write below stays deferred.
        byte[]? tapeBytes = save.PreFinishedRoom != null
            ? SimCombatUndo.SerializeLiveTape()
            : null;
        RunState? state = RunManager.Instance.IsInProgress
            ? RunManager.Instance.DebugOnlyGetState()
            : null;

        SimSnapshotMeta meta;
        lock (Lock)
        {
            if (save.StartTime != _currentStartTime)
            {
                BeginRun(save.StartTime);
            }
            meta = new SimSnapshotMeta
            {
                Seq = _nextSeq++,
                Ts = DateTime.UtcNow.ToString("o"),
                Floor = state?.TotalFloor ?? 0,
                Act = (state?.CurrentActIndex ?? save.CurrentActIndex) + 1,
                StartTime = save.StartTime,
                PostCombat = save.PreFinishedRoom != null,
                NumReloads = save.NumReloads,
                HasTape = tapeBytes != null,
            };
            meta.Id = $"{meta.StartTime}-{meta.Seq}";
            Metas.Add(meta);
            JsonById[meta.Id] = json;
            while (JsonById.Count > MaxInMemory)
            {
                string? oldest = Metas
                    .Where(m => JsonById.ContainsKey(m.Id))
                    .Select(m => m.Id)
                    .FirstOrDefault();
                if (oldest == null)
                {
                    break;
                }
                JsonById.Remove(oldest);
            }
        }

        // Disk write off the caller's context; the in-memory copy is already
        // authoritative for this session.
        Task.Run(() => Persist(meta, json, tapeBytes));

        BridgeServer.Instance.SendState(JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["type"] = "snapshot_saved",
            ["snapshot_id"] = meta.Id,
            ["seq"] = meta.Seq,
            ["floor"] = meta.Floor,
            ["act"] = meta.Act,
            ["post_combat"] = meta.PostCombat,
            ["has_tape"] = meta.HasTape,
            ["run_start_time"] = meta.StartTime,
            ["ts"] = meta.Ts,
        }));
        Logger.Log($"[SimSnapshots] #{meta.Seq} floor {meta.Floor} " +
                   (meta.PostCombat ? "(post-combat)" : "(room entry)"));
    }

    /// <summary>Switch runs: reload any snapshots persisted for this
    /// start_time (post-crash / post-restart rewind), else start fresh.</summary>
    private static void BeginRun(long startTime)
    {
        Metas.Clear();
        JsonById.Clear();
        _currentStartTime = startTime;
        _nextSeq = 0;
        try
        {
            string dir = RunDir(startTime);
            string indexPath = Path.Combine(dir, "index.json");
            if (File.Exists(indexPath))
            {
                var loaded = JsonSerializer.Deserialize<List<SimSnapshotMeta>>(
                    File.ReadAllText(indexPath));
                if (loaded is { Count: > 0 })
                {
                    Metas.AddRange(loaded.OrderBy(m => m.Seq));
                    _nextSeq = Metas[^1].Seq + 1;
                    Logger.Log($"[SimSnapshots] Resumed index for run {startTime}: " +
                               $"{Metas.Count} snapshots on disk.");
                }
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimSnapshots] Index reload failed: {ex.Message}");
        }
    }

    private static string RunDir(long startTime) =>
        Path.Combine(SnapshotsRoot, startTime.ToString());

    private static void Persist(SimSnapshotMeta meta, string json, byte[]? tapeBytes)
    {
        try
        {
            string dir = RunDir(meta.StartTime);
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, $"{meta.Seq:D4}_f{meta.Floor}.run.json"), json);
            if (tapeBytes != null)
            {
                File.WriteAllBytes(Path.Combine(dir, $"{meta.Seq:D4}_tape.mcr"), tapeBytes);
            }
            List<SimSnapshotMeta> metas;
            lock (Lock)
            {
                metas = Metas.Where(m => m.StartTime == meta.StartTime).ToList();
            }
            File.WriteAllText(
                Path.Combine(dir, "index.json"),
                JsonSerializer.Serialize(metas, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimSnapshots] Persist failed: {ex.Message}");
        }
    }

    // ------------------------------------------------------------------
    // Queries (any thread)
    // ------------------------------------------------------------------

    public static List<SimSnapshotMeta> List()
    {
        lock (Lock)
        {
            return Metas.ToList();
        }
    }

    public static long CurrentStartTime
    {
        get
        {
            lock (Lock)
            {
                return _currentStartTime;
            }
        }
    }

    /// <summary>The snapshot JSON, from memory or disk. Null if unknown.</summary>
    public static string? GetJson(string id)
    {
        SimSnapshotMeta? meta;
        lock (Lock)
        {
            if (JsonById.TryGetValue(id, out string? cached))
            {
                return cached;
            }
            meta = Metas.FirstOrDefault(m => m.Id == id);
        }
        if (meta == null)
        {
            return null;
        }
        try
        {
            string path = Path.Combine(RunDir(meta.StartTime), $"{meta.Seq:D4}_f{meta.Floor}.run.json");
            return File.Exists(path) ? File.ReadAllText(path) : null;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimSnapshots] Read failed for {id}: {ex.Message}");
            return null;
        }
    }

    /// <summary>The newest room-entry snapshot (the current room's start) --
    /// the "⟲ restart room" target. Null when nothing is captured yet.</summary>
    /// <summary>Path of the persisted combat tape for snapshot id
    /// <paramref name="id"/> (a post-combat snapshot), or null when none was
    /// captured / it is gone from disk.</summary>
    public static string? TapeFile(string id)
    {
        int dash = id.LastIndexOf('-');
        if (dash <= 0 ||
            !long.TryParse(id[..dash], out long start) ||
            !int.TryParse(id[(dash + 1)..], out int seq))
        {
            return null;
        }
        string path = Path.Combine(RunDir(start), $"{seq:D4}_tape.mcr");
        return File.Exists(path) ? path : null;
    }

    public static SimSnapshotMeta? CurrentRoomEntry()
    {
        lock (Lock)
        {
            return Metas.LastOrDefault(m => !m.PostCombat);
        }
    }

    /// <summary>The room-entry snapshot before the current one -- the
    /// "⟲ back one room" target.</summary>
    public static SimSnapshotMeta? PreviousRoomEntry()
    {
        lock (Lock)
        {
            var entries = Metas.Where(m => !m.PostCombat).ToList();
            return entries.Count >= 2 ? entries[^2] : null;
        }
    }
}
