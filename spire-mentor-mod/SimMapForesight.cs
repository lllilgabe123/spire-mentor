// SimMapForesight.cs -- what is waiting along the map (personal build).
//
// Rooms are not random when you enter them in the way they look. Each act
// keeps fixed queues (RoomSet: normalEncounters / eliteEncounters / events
// plus a visited counter for each), and a "?" room is one NextFloat() from
// the run's UnknownMapPoint stream against UnknownMapPointOdds
// (Odds/UnknownMapPointOdds.cs Roll, called from RunManager.RollRoomTypeFor
// with BuildRoomTypeBlacklist). So walking a route in order and replaying
// those rules on COPIES gives the rooms exactly:
//
//   ?        -> Event (which one) / Monster (which fight) / Shop / Treasure
//   Monster  -> which fight     Elite -> which elite     Boss -> which boss
//
// Copies only: the stream is cloned through ToSerializable (a read), the
// odds and counters are copied into locals, and the hooks it consults
// (ModifyUnknownMapPointRoomTypes, ModifyOddsIncreaseForUnrolledRoomType:
// Juzu Bracelet, Golden Compass, Lantern Key, Deadly Events) only compute.
// Nothing in the game moves, so co-op games stay in sync.
//
// Limits: an event that would be skipped because of what you have when you
// arrive (EventModel.IsAllowed) is judged with what you have now, and relics
// picked up on the way are not foreseen.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Odds;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal static class SimMapForesight
{
    private const BindingFlags Any = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    private static string? _lastLogged;

    /// <summary>Fill <see cref="SimAdvice.PreviewLines"/> for map advice:
    /// the recommended route's rooms, then what each next choice holds.</summary>
    public static void Apply(SimAdvice advice)
    {
        try
        {
            RunState? rs = RunManager.Instance.DebugOnlyGetState();
            if (rs == null || rs.UnlockState.NumberOfRuns == 0)
            {
                return; // the first-ever run's "?" rooms follow the tutorial script
            }
            if (advice.Path.Count > 0)
            {
                List<MapPoint> route = advice.Path
                    .Select(n => rs.Map.GetPoint(n.Col, n.Row))
                    .TakeWhile(p => p != null)
                    .Cast<MapPoint>()
                    .ToList();
                if (route.Count > 0)
                {
                    advice.PreviewLines.Add("Route ahead: " + string.Join(" > ", Walk(rs, route)));
                }
            }
            IEnumerable<MapPoint> next = rs.CurrentMapPoint?.Children ?? rs.Map.GetPointsInRow(0);
            List<string> choices = next.OrderBy(p => p.coord.col)
                .Select(p => Walk(rs, new List<MapPoint> { p })[0])
                .ToList();
            if (choices.Count > 1)
            {
                advice.PreviewLines.Add("Next choices (left to right): " + string.Join(" | ", choices));
            }
            string joined = string.Join(" / ", advice.PreviewLines);
            if (joined.Length > 0 && joined != _lastLogged)
            {
                _lastLogged = joined;
                Logger.Log($"[SimMapForesight] {joined}");
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimMapForesight] skipped: {ex.Message}");
        }
    }

    /// <summary>The rooms <paramref name="route"/> would give, entered in order
    /// from where the run is now, replayed on copies.</summary>
    private static List<string> Walk(RunState rs, List<MapPoint> route)
    {
        UnknownMapPointOdds odds = rs.Odds.UnknownMapPoint;
        Rng rng = new Rng(((Rng)Field(typeof(AbstractOdds), odds, "_rng")).ToSerializable());
        var nonEvent = new Dictionary<RoomType, float>((Dictionary<RoomType, float>)Field(typeof(UnknownMapPointOdds), odds, "_nonEventOdds"));
        var baseOdds = (Dictionary<RoomType, float>)Field(typeof(UnknownMapPointOdds), odds, "_baseOdds");
        var rooms = (RoomSet)Field(typeof(ActModel), rs.Act, "_rooms");
        int normal = rooms.normalEncountersVisited;
        int elite = rooms.eliteEncountersVisited;
        int ev = rooms.eventsVisited;
        var seen = new HashSet<ModelId>(rs.VisitedEventIds);
        bool prevShop = rs.CurrentMapPointHistoryEntry?.HasRoomOfType(RoomType.Shop) ?? false;

        var out_ = new List<string>();
        foreach (MapPoint p in route)
        {
            RoomType room = p.PointType switch
            {
                MapPointType.Unknown => Roll(rs, rng, nonEvent, baseOdds,
                    prevShop || (p.Children.Count > 0 && p.Children.All(c => c.PointType == MapPointType.Shop))),
                MapPointType.Shop => RoomType.Shop,
                MapPointType.Treasure => RoomType.Treasure,
                MapPointType.RestSite => RoomType.RestSite,
                MapPointType.Monster => RoomType.Monster,
                MapPointType.Elite => RoomType.Elite,
                MapPointType.Boss => RoomType.Boss,
                MapPointType.Ancient => RoomType.Event,
                _ => RoomType.Unassigned,
            };
            string prefix = p.PointType == MapPointType.Unknown ? "? = " : "";
            string what = room switch
            {
                RoomType.Monster when rooms.normalEncounters.Count > 0 =>
                    "Fight: " + Name(rooms.normalEncounters[normal++ % rooms.normalEncounters.Count]),
                RoomType.Elite when rooms.eliteEncounters.Count > 0 =>
                    "ELITE: " + Name(rooms.eliteEncounters[elite++ % rooms.eliteEncounters.Count]),
                RoomType.Boss => "BOSS: " + Name(rooms.Boss),
                RoomType.Event when p.PointType == MapPointType.Ancient => "Ancient",
                RoomType.Event => "Event: " + NextEvent(rs, rooms, ref ev, seen),
                RoomType.RestSite => "Rest",
                RoomType.Treasure => "Treasure",
                RoomType.Shop => "Shop",
                _ => room.ToString(),
            };
            out_.Add(prefix + what);
            prevShop = room == RoomType.Shop;
        }
        return out_;
    }

    /// <summary>UnknownMapPointOdds.Roll on copies (the tutorial branch is
    /// excluded by the caller).</summary>
    private static RoomType Roll(RunState rs, Rng rng, Dictionary<RoomType, float> nonEvent,
        Dictionary<RoomType, float> baseOdds, bool noShop)
    {
        var allowed = nonEvent.Keys.Append(RoomType.Event).Where(t => !(noShop && t == RoomType.Shop)).ToHashSet();
        IReadOnlySet<RoomType> types = Hook.ModifyUnknownMapPointRoomTypes(rs, allowed);
        RoomType room = types.Contains(RoomType.Event) ? RoomType.Event : types.Order().First();
        float roll = rng.NextFloat();
        float acc = 0f;
        foreach ((RoomType t, float odds) in nonEvent)
        {
            if (types.Contains(t) && !(odds < 0f))
            {
                acc += odds;
                if (roll <= acc)
                {
                    room = t;
                    break;
                }
            }
        }
        foreach ((RoomType t, float b) in baseOdds)
        {
            if (room == t)
            {
                nonEvent[t] = b;
            }
            else if (types.Contains(t))
            {
                nonEvent[t] += Hook.ModifyOddsIncreaseForUnrolledRoomType(rs, t, b);
            }
        }
        return room;
    }

    /// <summary>RoomSet.EnsureNextEventIsValid + NextEvent, on copies.</summary>
    private static string NextEvent(RunState rs, RoomSet rooms, ref int ev, HashSet<ModelId> seen)
    {
        if (rooms.events.Count == 0)
        {
            return "?";
        }
        for (int i = 0; i < rooms.events.Count; i++)
        {
            EventModel e = rooms.events[ev % rooms.events.Count];
            if (e.IsAllowed(rs) && !seen.Contains(e.Id))
            {
                break;
            }
            ev++;
        }
        EventModel pick = rooms.events[ev % rooms.events.Count];
        seen.Add(pick.Id);
        ev++;
        return Name(pick);
    }

    private static string Name(AbstractModel m)
    {
        try
        {
            string? t = m switch
            {
                EncounterModel enc => enc.Title.GetFormattedText(),
                EventModel e => e.Title.GetFormattedText(),
                _ => null,
            };
            if (!string.IsNullOrWhiteSpace(t))
            {
                return t;
            }
        }
        catch
        {
            // fall back to the id
        }
        return m.Id.Entry;
    }

    private static object Field(Type t, object o, string name) =>
        (t.GetField(name, Any) ?? throw new MissingFieldException(t.Name, name)).GetValue(o)
        ?? throw new InvalidOperationException($"{t.Name}.{name} is null");
}
