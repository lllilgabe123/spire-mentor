// SimCoop.cs -- teammate context for co-op fight advice (personal build).
//
// The engine simulates one player, so in a co-op fight the search believes
// only your cards hurt the enemies and plans for a fight several turns longer
// than the real one (the reliability tracker's co-op runs: 47% average win
// estimate against 8/8 fights won). This adds a `coop` block to each combat
// advice request:
//
//   teammate_share  the share of this fight's damage your living teammates
//                   have dealt (SimWatchdog's tally), falling back to this
//                   run's finished fights and then to a cautious prior while
//                   the fight is young. The advisor scales enemy HP by
//                   1 - share for its search (realgame_foresight.py).
//   teammates       each living teammate's board, built by the same
//                   BuildCombatState as yours, so the advisor can suggest
//                   their turn for you to call out.
//
// Read-only: everything comes from your own game's state and history, and it
// only goes to the advisor on your PC. Solo requests carry no `coop` block.

using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal static class SimCoop
{
    /// <summary>Damage dealt this fight before its own share is trusted fully.</summary>
    private const int FightTrust = 40;

    /// <summary>Damage over the run's finished fights before the run share
    /// replaces the prior.</summary>
    private const int RunTrust = 100;

    /// <summary>Prior teammate share per teammate slot, before any history:
    /// an even split shaded down, so a quiet teammate cannot make the advice
    /// reckless.</summary>
    private const double PriorShade = 0.8;

    public static void Attach(Dictionary<string, object> state)
    {
        try
        {
            if (!SimMultiplayer.InRun)
            {
                return;
            }
            RunState? run = RunManager.Instance.DebugOnlyGetState();
            Player? me = run != null ? LocalContext.GetMe(run) : null;
            if (run == null || me == null)
            {
                return;
            }
            List<Player> mates = run.Players
                .Where(p => p.NetId != me.NetId && p.Creature != null && !p.Creature.IsDead)
                .ToList();
            double share = mates.Count == 0 ? 0.0 : Share(me, mates, run.Players.Count);

            var boards = new List<Dictionary<string, object?>>();
            foreach (Player mate in mates)
            {
                Dictionary<string, object>? board = SimCombatHandler.BuildEmbeddedCombatState(mate);
                if (board != null)
                {
                    boards.Add(new Dictionary<string, object?>
                    {
                        ["name"] = Name(mate),
                        ["board"] = board,
                    });
                }
            }
            state["coop"] = new Dictionary<string, object?>
            {
                ["players"] = run.Players.Count,
                ["teammate_share"] = Math.Round(share, 3),
                ["teammates"] = boards,
            };
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCoop] skipped: {ex.Message}");
        }
    }

    private static double Share(Player me, List<Player> mates, int players)
    {
        HashSet<ulong> mateIds = mates.Select(p => p.NetId).ToHashSet();
        double prior = PriorShade * mates.Count / Math.Max(1, players);

        (int mine, int theirs) Split(IReadOnlyDictionary<ulong, int>? d) => d == null
            ? (0, 0)
            : (d.GetValueOrDefault(me.NetId), d.Where(kv => mateIds.Contains(kv.Key)).Sum(kv => kv.Value));

        (int runMine, int runTheirs) = Split(SimWatchdog.RunDamage);
        double fallback = runMine + runTheirs >= RunTrust ? (double)runTheirs / (runMine + runTheirs) : prior;

        (int mine, int theirs) = Split(SimWatchdog.FightDamage);
        int total = mine + theirs;
        if (total <= 0)
        {
            return fallback;
        }
        double w = Math.Min(1.0, (double)total / FightTrust);
        return w * theirs / total + (1 - w) * fallback;
    }

    private static string Name(Player p)
    {
        string? id = p.Character?.Id.Entry;
        return string.IsNullOrEmpty(id)
            ? "Teammate"
            : "teammate (" + char.ToUpperInvariant(id[0]) + id.Substring(1).ToLowerInvariant() + ")";
    }
}
