// SimMp.cs -- co-op multiplayer awareness (docs/plans/2026-10-23-coop-multiplayer.md).
//
// The mod reuses the single-player advisor in a co-op run: it plans for the
// local player only. What it must never do there is change game state outside
// the synced action queue (undo, restore, autopilot, a direct PlayerCmd), so
// every such path asks IsMultiplayer first.
//
// The command-line overrides exist for the two-instance test lane
// (tools/coop_test.ps1): both game instances share one SpireMentor.conf, so
// the dummy instance's port and autoplay come from its own command line. The
// game parses its arguments the same way (CommandLineHelper: `--key value` or
// `--key=value`).

using System;
using Godot;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

public static class SimMp
{
    /// <summary>
    /// A run with other players over the network, as host or client. Replay
    /// and single-player are not; neither is the main menu (no NetService).
    /// </summary>
    public static bool IsMultiplayer
    {
        get
        {
            try
            {
                NetGameType? type = RunManager.Instance?.NetService?.Type;
                return type == NetGameType.Host || type == NetGameType.Client;
            }
            catch (Exception)
            {
                return false;
            }
        }
    }

    /// <summary>Players in the current run (1 outside a run).</summary>
    public static int PlayerCount
    {
        get
        {
            try
            {
                return Math.Max(1, RunManager.Instance?.DebugOnlyGetState()?.Players.Count ?? 1);
            }
            catch (Exception)
            {
                return 1;
            }
        }
    }

    /// <summary>
    /// A game command's arguments name a player other than us. Every peer
    /// runs every player's card effects, so a hook on such a command sees
    /// teammates' choices too.
    /// </summary>
    public static bool IsAnotherPlayers(object?[]? args)
    {
        if (args == null)
        {
            return false;
        }
        foreach (object? a in args)
        {
            if (a is Player p)
            {
                return !LocalContext.IsMe(p);
            }
        }
        return false;
    }

    // The "already voted" checks below. In co-op the map, a shared event and
    // the treasure relics are votes: the screen stays up, still clickable,
    // until every player has voted, so autoplay would keep re-pressing (and
    // its looping-event escape would press a DIFFERENT option, i.e. change
    // the vote). Once ours is in, the screen is not a decision for us.
    // Each synchronizer clears its votes when the vote resolves.

    /// <summary>Our vote for the next map node is in (for this map).</summary>
    public static bool HasVotedOnMap(Player? me)
    {
        if (me == null || !IsMultiplayer)
        {
            return false;
        }
        try
        {
            MapSelectionSynchronizer sync = RunManager.Instance.MapSelectionSynchronizer;
            return sync.GetVote(me) is MapVote vote && vote.mapGenerationCount == sync.MapGenerationCount;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>A shared event's page is up and our option vote is in.</summary>
    public static bool HasVotedOnEvent(Player? me)
    {
        if (me == null || !IsMultiplayer)
        {
            return false;
        }
        try
        {
            EventSynchronizer sync = RunManager.Instance.EventSynchronizer;
            return sync.IsShared && sync.GetPlayerVote(me).HasValue;
        }
        catch (Exception)
        {
            // IsShared throws when no event is in progress.
            return false;
        }
    }

    /// <summary>Our treasure-relic vote (or skip) is in.</summary>
    public static bool HasVotedOnTreasure(Player? me)
    {
        if (me == null || !IsMultiplayer)
        {
            return false;
        }
        try
        {
            return RunManager.Instance.TreasureRoomRelicSynchronizer.GetPlayerVote(me)?.voteReceived == true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>The toast for a feature refused in co-op.</summary>
    public static string RefusedMessage(string feature) => $"{feature} is not available in co-op.";

    /// <summary><c>--spire-mentor-port N</c>: this instance's bridge port.</summary>
    public static int? PortOverride => int.TryParse(CmdlineValue("spire-mentor-port"), out int p) ? p : null;

    /// <summary><c>--spire-mentor-autoplay</c>: arm advisor autoplay on start
    /// (the dummy co-op partner).</summary>
    public static bool AutoplayOverride => HasCmdlineArg("spire-mentor-autoplay");

    /// <summary><c>--spire-mentor-speed X</c>: game speed for this launch
    /// (SimSpeed, not persisted).</summary>
    public static double? SpeedOverride =>
        double.TryParse(CmdlineValue("spire-mentor-speed"), System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out double s) ? s : null;

    /// <summary><c>--spire-mentor-no-saves</c>: write no saves (SimNoSaves);
    /// the dummy partner shares the user's Steam profile.</summary>
    public static bool NoSavesOverride => HasCmdlineArg("spire-mentor-no-saves");

    /// <summary><c>--spire-mentor-character NAME</c>: the dummy's character
    /// in the co-op lobby (SimCoopDummy).</summary>
    public static string? CharacterOverride => CmdlineValue("spire-mentor-character");

    /// <summary>The game's own <c>--fastmp</c> dev launch (host / join a
    /// local co-op lobby without Steam).</summary>
    public static bool FastMpLaunch => HasCmdlineArg("fastmp");

    private static bool HasCmdlineArg(string key) => FindCmdlineArg(key, out _);

    private static string? CmdlineValue(string key) => FindCmdlineArg(key, out string? v) ? v : null;

    private static bool FindCmdlineArg(string key, out string? value)
    {
        value = null;
        string[] args;
        try
        {
            args = OS.GetCmdlineArgs();
        }
        catch (Exception)
        {
            return false;
        }
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i].TrimStart('-');
            if (a == key)
            {
                if (i + 1 < args.Length && !args[i + 1].StartsWith('-'))
                {
                    value = args[i + 1];
                }
                return true;
            }
            if (a.StartsWith(key + "=", StringComparison.Ordinal))
            {
                value = a.Substring(key.Length + 1);
                return true;
            }
        }
        return false;
    }
}
