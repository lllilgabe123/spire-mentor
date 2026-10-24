// SimWorkbenchLink.cs -- the toolbar's bridge to the workbench web app.
//
// The workbench is the mod's "separate window": logs, search trees and the
// run/branch tree render in a browser instead of ever-growing in-game
// overlays. Two of them can answer, tried in this order:
//   1. the BUNDLED one -- the browser-only build the mod zip ships in
//      mods/SpireMentor/workbench/, hosted by the advisor's control plane
//      on :8079 (sts2-pycli/python/realgame_control.py). Up while the AI
//      coach is on; the page is connected to the live game from the start,
//      and `?follow=1` follows the advisor's run log there too.
//   2. the DEV one -- the Python workbench (FastAPI on :8077), which the
//      advisor's `--workbench auto` spawns in dev installs.
// The mod's only jobs are (a) know the URL and (b) open the default browser
// on it; starting either server is the advisor's business, never the mod's.

using System;
using System.Net.Http;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace SpireMentor;

internal static class SimWorkbenchLink
{
    /// <summary>The bundled workbench, on the advisor's control plane (its
    /// default --control-port), following the live run log like the dev one.</summary>
    private const string BundledUrl = "http://127.0.0.1:8079/?follow=1";

    /// <summary>The dev workbench, following the advisor's live run log.</summary>
    private const string DevUrl = "http://127.0.0.1:8077/?follow=1";

    /// <summary>
    /// Open the workbench in the default browser. Probes the server first
    /// (off the main thread) so a dead workbench produces a helpful note
    /// instead of a browser error page. Never throws, never blocks.
    /// <paramref name="run"/> / <paramref name="step"/> (a coach record's
    /// <c>log_name</c> / <c>log_idx</c>) open the replay at that decision
    /// instead of the newest step.
    /// </summary>
    public static void Open(string? run = null, int? step = null) => TaskHelper.RunSafely(OpenAsync(run, step));

    private static async Task OpenAsync(string? run, int? step)
    {
        // A configured URL is the only candidate (SpireMentorModConfig.WorkbenchUrl).
        string configured = SpireMentorModConfig.WorkbenchUrl.Trim();
        string[] candidates = configured.Length > 0 ? new[] { configured } : new[] { BundledUrl, DevUrl };
        string? url = null;
        foreach (string candidate in candidates)
        {
            if (await ProbeAsync(candidate))
            {
                url = candidate;
                break;
            }
        }
        if (url == null)
        {
            SimActionLog.Note(configured.Length > 0
                ? $"Workbench is not reachable at {configured} (the Workbench URL setting)."
                : "Workbench is not reachable — turn the AI coach on (Insert): the advisor hosts it. " +
                  "In a dev install, `just workbench` in sts2-pycli starts the Python one.");
            Logger.Log($"[SimWorkbenchLink] Health probe failed for {string.Join(", ", candidates)}; not opening a browser.");
            return;
        }
        url = WorkbenchUrl.WithFollowParams(url, run, step);
        // NOT OS.ShellOpen: that spawns the browser as a CHILD OF THE GAME,
        // and Steam waits for the game's whole descendant tree on quit -- a
        // browser opened this way left Steam stuck on "closing" until the
        // browser exited (observed 2026-08-21, Firefox). explorer.exe
        // delegates to the long-lived shell instead -- but it cannot parse a
        // raw URL with a query string (it opens a folder), so the URL goes
        // through an InternetShortcut file, which the shell resolves to the
        // default browser with the query intact.
        try
        {
            string dir = System.IO.Path.Combine(
                System.Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData),
                "SpireMentor");
            System.IO.Directory.CreateDirectory(dir);
            string shortcut = System.IO.Path.Combine(dir, "workbench.url");
            System.IO.File.WriteAllText(shortcut,
                "[InternetShortcut]\r\nURL=" + url + "\r\n");
            using var p = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = "\"" + shortcut + "\"",
                UseShellExecute = false,
            });
            SimActionLog.Note($"Opened workbench: {url}");
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimWorkbenchLink] Open failed: {ex.Message}");
            SimActionLog.Note($"Could not open the browser. Workbench URL: {url}");
        }
    }

    private static async Task<bool> ProbeAsync(string url)
    {
        try
        {
            var baseUri = new Uri(url);
            var health = new Uri($"{baseUri.Scheme}://{baseUri.Authority}/api/health");
            using var http = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromMilliseconds(1500) };
            using HttpResponseMessage resp = await http.GetAsync(health);
            return resp.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
}
