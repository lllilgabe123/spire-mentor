// SimNoSaves.cs -- `--spire-mentor-no-saves`: this instance never writes saves.
//
// The co-op test lane (tools/coop_test.ps1) runs two game instances under ONE
// Steam account, so both read and write the same profile: progress.save,
// prefs, the co-op run save, the history. Two writers race on the game's
// write-tmp-then-rename (a "Failed to rename file" SaveException was logged
// on the first test run, 2026-10-23), and the profile is the user's real
// one. The dummy partner is throwaway, so it simply does not write: every
// mutating call on both save stores (local files and Steam Cloud) becomes a
// no-op. Reads are untouched, so it still starts from the real profile.

using System;
using System.Reflection;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Platform.Steam;
using MegaCrit.Sts2.Core.Saves;

namespace SpireMentor;

internal static class SimNoSaves
{
    private static readonly string[] MutatingMethods =
    {
        "WriteFile", "WriteFileAsync", "DeleteFile", "RenameFile",
        "SetLastModifiedTime", "DeleteDirectory", "DeleteTemporaryFiles",
        // SteamRemoteStorage.FileForget: drops the file from Steam Cloud and
        // keeps only the local copy, so the user's cloud save is lost all the
        // same. CloudSaveStore calls it during sync.
        "ForgetFile",
    };

    public static void InstallIfRequested(Harmony harmony)
    {
        if (!SimMp.NoSavesOverride)
        {
            return;
        }
        int patched = 0;
        foreach (Type store in new[] { typeof(GodotFileIo), typeof(SteamRemoteSaveStore) })
        {
            foreach (MethodInfo m in store.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
            {
                if (Array.IndexOf(MutatingMethods, m.Name) < 0)
                {
                    continue;
                }
                string prefix = m.ReturnType == typeof(void) ? nameof(SkipVoid)
                    : m.ReturnType == typeof(Task) ? nameof(SkipTask)
                    : "";
                if (prefix == "")
                {
                    Logger.Log($"[SimNoSaves] Not patching {store.Name}.{m.Name}: returns {m.ReturnType.Name}.");
                    continue;
                }
                harmony.Patch(m, prefix: new HarmonyMethod(typeof(SimNoSaves), prefix));
                patched++;
            }
        }
        Logger.Log($"[SimNoSaves] --spire-mentor-no-saves: {patched} save-store writes disabled; this instance leaves the profile alone.");
    }

    private static bool SkipVoid() => false;

    private static bool SkipTask(ref Task __result)
    {
        __result = Task.CompletedTask;
        return false;
    }
}
