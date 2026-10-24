// SimRouteRisk.cs -- warn when the recommended map route is a gamble.
//
// Map advice carries the run AI's recommended route with the HP it expects
// on arrival at each node (SimAdvice.Path, `hp`). It never said when that
// forecast was grim: in one live A4 run it predicted, six floors ahead, an
// Elite at 34/80 and arrival at the last rest site with 12 HP -- and the run
// then went into that Elite at 41, left it at 10, and met the boss at 34 HP
// with a 2% fight estimate. Replayed over every logged map decision
// (tools/route_risk_replay.py), these rules flagged that run from floor 8
// on, stayed silent on the runs that reached the boss healthy (94-100%), and
// fire on ~15% of map decisions overall.
//
// Warns (display only) when the recommended route
//   1. enters an Elite below EliteFraction of max HP, or
//   2. reaches the boss below BossFraction of max HP, counting the 30% heal
//      (HealRestSiteOption.GetBaseHealAmount) of a final rest site.
// In co-op the forecast, like the fight AI, ignores the teammate, so the
// warning says it is pessimistic there.

using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal static class SimRouteRisk
{
    public const double EliteFraction = 0.5;
    public const double BossFraction = 0.5;
    private const double RestHeal = 0.3;

    private static string? _lastLogged;

    /// <summary>Set <see cref="SimAdvice.RiskNote"/> on map advice whose
    /// route forecast is risky.</summary>
    public static void Apply(SimAdvice advice, Player? me)
    {
        try
        {
            if (me == null || advice.Path.Count == 0)
            {
                return;
            }
            int max = me.Creature.MaxHp;
            if (max <= 0)
            {
                return;
            }
            var parts = new List<string>();
            for (int i = 0; i < advice.Path.Count; i++)
            {
                SimPathNode n = advice.Path[i];
                if (n.Type == "Elite" && n.Hp is int hp && hp < EliteFraction * max)
                {
                    string when = i == 0 ? "next" : $"{i + 1} floors ahead";
                    parts.Add($"Elite {when} at ~{hp}/{max} HP");
                }
            }
            SimPathNode last = advice.Path[^1];
            if (last.Hp is int endHp)
            {
                double atBoss = endHp + (last.Type == "RestSite" ? RestHeal * max : 0);
                if (atBoss < BossFraction * max)
                {
                    parts.Add($"reaches the boss with ~{atBoss:F0}/{max} HP");
                }
            }
            if (parts.Count == 0)
            {
                return;
            }
            bool coop = RunManager.Instance.NetService?.Type.IsMultiplayer() ?? false;
            advice.RiskNote = "⚠ risky route: " + string.Join("; ", parts) +
                              (coop ? " (solo forecast: it ignores your teammate, so it is pessimistic)" : "");
            if (advice.RiskNote != _lastLogged)
            {
                _lastLogged = advice.RiskNote;
                Logger.Log($"[SimRouteRisk] {advice.RiskNote}");
                SimActionLog.Note(advice.RiskNote);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimRouteRisk] skipped: {ex.Message}");
        }
    }
}
