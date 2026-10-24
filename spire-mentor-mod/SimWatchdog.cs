// SimWatchdog.cs -- notice when the advisor cannot read the game, and note
// every run start, both locally.
//
// 1. An advice reply whose note starts "advisor error:" means the backend
//    could not even parse the position -- e.g. a v0.111.0 card its tables
//    did not resolve (`$.deck.hand[2]: MIDNIGHT`). Advice then just vanished
//    with nothing on screen. Now the first occurrence of each distinct
//    problem per session puts a warning on the toolbar and in the action
//    log, and every occurrence is appended to watchdog.jsonl.
// 2. Every run start appends one line to runs.jsonl (seed, character,
//    singleplayer / host / client, player count), so tools/tracker.py can
//    split its reliability report into solo and co-op runs -- the advisor's
//    own run logs do not say which a run was.
//
// 3. Every fight end appends one line to fights.jsonl: the damage each player
//    (pets counted for their owner) dealt to enemies, read from this game's
//    own CombatHistory -- how much of a co-op fight the teammate carries,
//    which the single-player fight engine cannot see (tools/tracker.py).
//
// Client-side only: all three write files under %LOCALAPPDATA%\SpireMentor
// and draw on this machine's HUD; nothing is sent to the game or to other
// players, and the history is only read.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal static class SimWatchdog
{
    private const string ErrorPrefix = "advisor error:";
    private static readonly HashSet<string> _reported = new();
    private static readonly object _fileLock = new();

    public static void Install()
    {
        RunManager.Instance.RunStarted += OnRunStarted;
    }

    /// <summary>Called with every advice reply's note (SimAdvisor).</summary>
    public static void OnAdviceNote(string? note)
    {
        if (note == null || !note.StartsWith(ErrorPrefix, StringComparison.Ordinal))
        {
            return;
        }
        try
        {
            string detail = note.Substring(ErrorPrefix.Length).Trim();
            // `$.deck.hand[2]: MIDNIGHT` -> MIDNIGHT; anything else as is.
            Match m = Regex.Match(detail, @":\s*([A-Z0-9_]+)\s*$");
            string what = m.Success ? m.Groups[1].Value : detail;
            Append("watchdog.jsonl", new Dictionary<string, object?>
            {
                ["time"] = DateTime.Now.ToString("s"),
                ["game_version"] = SimGameVersion.Wire,
                ["unreadable"] = what,
                ["detail"] = detail,
            });
            bool first;
            lock (_reported)
            {
                first = _reported.Add(what);
            }
            if (!first)
            {
                return;
            }
            Logger.Log($"[SimWatchdog] The advisor cannot read this position ({detail}); advice paused while it lasts.");
            SimActionLog.Note($"AI can't read {what} -- advice is paused while it is in play. Details: {detail}");
            SimToolbar.Instance.SetNote($"⚠ AI can't read {what}: advice paused");
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimWatchdog] note handling failed: {ex.Message}");
        }
    }

    private static bool _combatHooked;

    // The fight's damage so far. The game clears CombatHistory BEFORE it
    // raises CombatEnded (CombatManager.cs:1319 vs :1361), so the tally is
    // kept up to date on every history change and only written at the end.
    private static Dictionary<ulong, int>? _dealt;
    private static int _otherDealt;

    // Every finished fight of this run, per player (SimCoop's fallback when
    // the current fight has too little history to judge a teammate by).
    private static readonly Dictionary<ulong, int> _runDealt = new();

    /// <summary>This fight's damage per player so far (null before the first hit).</summary>
    internal static IReadOnlyDictionary<ulong, int>? FightDamage => _dealt;

    /// <summary>Damage per player over this run's finished fights.</summary>
    internal static IReadOnlyDictionary<ulong, int> RunDamage => _runDealt;

    private static void OnHistoryChanged()
    {
        try
        {
            RunState? state = RunManager.Instance.DebugOnlyGetState();
            List<DamageReceivedEntry> hits = CombatManager.Instance.History.Entries.OfType<DamageReceivedEntry>().ToList();
            if (state == null || hits.Count == 0)
            {
                return; // the end-of-fight Clear(): keep the last tally
            }
            var dealt = state.Players.ToDictionary(p => p.NetId, _ => 0);
            int other = 0;
            foreach (DamageReceivedEntry e in hits)
            {
                DamageResult r = e.Result;
                if (r.Receiver == null || !r.Receiver.IsMonster)
                {
                    continue;
                }
                // Block stripped counts as work done; overkill does not.
                int amount = r.BlockedDamage + Math.Max(0, r.UnblockedDamage - r.OverkillDamage);
                Player? owner = e.Dealer?.Player ?? e.Dealer?.PetOwner;
                if (owner != null && dealt.ContainsKey(owner.NetId))
                {
                    dealt[owner.NetId] += amount;
                }
                else
                {
                    other += amount;
                }
            }
            _dealt = dealt;
            _otherDealt = other;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimWatchdog] damage tally failed: {ex.Message}");
        }
    }

    private static void OnCombatEnded(CombatRoom? room)
    {
        try
        {
            RunState? state = RunManager.Instance.DebugOnlyGetState();
            Dictionary<ulong, int>? dealt = _dealt;
            int other = _otherDealt;
            _dealt = null;
            _otherDealt = 0;
            if (state == null || dealt == null)
            {
                return;
            }
            foreach (KeyValuePair<ulong, int> kv in dealt)
            {
                _runDealt[kv.Key] = _runDealt.GetValueOrDefault(kv.Key) + kv.Value;
            }
            Player? me = LocalContext.GetMe(state);
            Append("fights.jsonl", new Dictionary<string, object?>
            {
                ["time"] = DateTime.Now.ToString("s"),
                ["seed"] = state.Rng.StringSeed,
                ["floor"] = state.TotalFloor,
                ["room"] = room?.RoomType.ToString(),
                ["rounds"] = room?.CombatState?.RoundNumber,
                ["net"] = RunManager.Instance.NetService?.Type.ToString(),
                ["me"] = me?.NetId.ToString(),
                ["damage"] = dealt.ToDictionary(kv => kv.Key.ToString(), kv => kv.Value),
                ["other_damage"] = other,
            });
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimWatchdog] fight note failed: {ex.Message}");
        }
    }

    private static void OnRunStarted(RunState state)
    {
        if (!_combatHooked)
        {
            _combatHooked = true;
            CombatManager.Instance.History.Changed += OnHistoryChanged;
            CombatManager.Instance.CombatEnded += OnCombatEnded;
        }
        _runDealt.Clear();
        try
        {
            RunManager rm = RunManager.Instance;
            Player? me = LocalContext.GetMe(state);
            Append("runs.jsonl", new Dictionary<string, object?>
            {
                ["time"] = DateTime.Now.ToString("s"),
                ["seed"] = state.Rng.StringSeed,
                ["character"] = me?.Character?.Id.Entry,
                ["ascension"] = state.AscensionLevel,
                ["net"] = rm.NetService?.Type.ToString(),
                ["players"] = state.Players.Count,
                ["game_version"] = SimGameVersion.Wire,
            });
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimWatchdog] run note failed: {ex.Message}");
        }
    }

    private static void Append(string file, Dictionary<string, object?> row)
    {
        string path = Path.Combine(SimAdvisorProcess.AppDataDir(), file);
        string line = JsonSerializer.Serialize(row);
        lock (_fileLock)
        {
            File.AppendAllText(path, line + Environment.NewLine);
        }
    }
}
