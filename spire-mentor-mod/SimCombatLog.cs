// SimCombatLog.cs -- the running game's own combat log, on the wire.
//
// The game records every hit, block, power, move and card movement in
// `CombatManager.Instance.History` (MegaCrit.Sts2.Core.Combat.History), which
// is exactly what the workbench's combat-log panel shows for a SIMULATED run.
// This maps the real game's entries into the SAME row shape the Rust engine
// emits (`sts2_core::event_log` -> `RunManager.combat_log()`), so a live game
// followed in the workbench renders through the identical Python/React path
// with no second renderer to keep in sync.
//
// Polled, not subscribed: `SimAdvisor` already reads `History.Entries` for its
// settle heuristic, the list only grows within a fight, and a poll cannot leak
// a handler when a combat ends mid-teardown. `Drain()` returns everything
// recorded since the previous call and resets itself when the entry count
// SHRINKS -- the only way that happens is a new combat's fresh History.
//
// Every mapper is defensive: this is display data and a null-ref here would
// take the advice request down with it, so anything unexpected degrades to the
// entry's own `Description` string.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;

namespace SpireMentor;

internal static class SimCombatLog
{
    /// <summary>How many History entries have already been sent.</summary>
    private static int _sent;

    /// <summary>
    /// The move the acting monster is mid-way through, so its damage rows can
    /// carry "using Butt Move" the way the game's log does. C# hands a
    /// `CardSource` to `DamageReceivedEntry` for card damage but nothing for a
    /// monster move, so the move has to be remembered as the entries stream
    /// past -- the same nesting the Rust engine's log derives.
    /// </summary>
    private static string? _openMoveId;
    private static string? _openMoveMonster;

    /// <summary>`Rd 3 (Player turn): ...` -- the header C#
    /// `CombatHistoryEntry.HumanReadableString` prefixes every line with. The
    /// round and side are PRIVATE on the entry, so this is the only way to read
    /// the values the entry itself was stamped with (rather than the values the
    /// board happens to hold when we drain).</summary>
    private static readonly Regex HeaderRe = new(
        @"^Rd (\d+) \((\w+) turn\)", RegexOptions.Compiled);

    /// <summary>
    /// `player_choice` rows not yet sent, each with the History entry count
    /// at the moment of the choice: Drain puts a row before the first entry
    /// recorded after it. Choices are not History entries; they arrive by
    /// event (<see cref="RecordChoice"/>), possibly off the main thread.
    /// </summary>
    private static readonly List<(int At, Dictionary<string, object?> Row)> _choices = new();

    private static readonly object _choicesLock = new();

    /// <summary>Forget everything (a new run / a restored snapshot).</summary>
    internal static void Reset()
    {
        _sent = 0;
        _openMoveId = null;
        _openMoveMonster = null;
        lock (_choicesLock)
        {
            // A choice made before the new combat's History began (an
            // opening Toolbox pick) still belongs to it.
            for (int i = 0; i < _choices.Count; i++)
            {
                _choices[i] = (0, _choices[i].Row);
            }
        }
    }

    /// <summary>
    /// A player's choice mid-combat, as the game syncs it between peers (C#
    /// `PlayerChoiceSynchronizer.PlayerChoiceReceived`, raised for ours and
    /// for every teammate's): who chose, the per-player choice id, the
    /// result's type (`CombatCard`, `CanonicalCard`, `Index`, ...) and what
    /// was picked, cards as `{id, upgraded}` in the order chosen. Lets a
    /// recorded co-op fight replay every seat's picks. Never throws.
    /// </summary>
    internal static void RecordChoice(
        string? owner, uint choiceId, string type,
        List<Dictionary<string, object?>> cards, List<int>? indexes, string? player)
    {
        try
        {
            if (CombatManager.Instance?.IsInProgress != true)
            {
                // A rest-site Smith or an event pick: not the fight's.
                return;
            }
            CombatState? state = CombatManager.Instance.DebugOnlyGetState();
            int at = CombatManager.Instance?.History?.Entries?.Count() ?? 0;
            var row = new Dictionary<string, object?>
            {
                ["round"] = state?.RoundNumber,
                ["side"] = state == null ? null : state.CurrentSide == CombatSide.Player ? "player" : "enemy",
                ["text"] = cards.Count > 0
                    ? $"{owner} chose {string.Join(", ", cards.Select(c => c["id"]))}"
                    : $"{owner} chose {type} {string.Join(", ", indexes ?? new List<int>())}{player}",
                ["source"] = null,
                ["kind"] = "player_choice",
                ["owner"] = owner,
                ["choice_id"] = choiceId,
                ["choice_type"] = type,
                ["cards"] = cards,
                ["indexes"] = indexes,
                ["player"] = player,
            };
            lock (_choicesLock)
            {
                _choices.Add((at, row));
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombatLog] RecordChoice failed: {ex.Message}");
        }
    }

    /// <summary>A card as a `player_choice` row names it.</summary>
    internal static Dictionary<string, object?> ChoiceCard(string? id, bool upgraded) =>
        new()
        {
            ["id"] = id,
            ["upgraded"] = upgraded,
        };

    /// <summary>The buffered choice rows made before History entry
    /// <paramref name="before"/>, removed from the buffer, oldest first.</summary>
    private static List<Dictionary<string, object?>> TakeChoices(int before)
    {
        var rows = new List<Dictionary<string, object?>>();
        lock (_choicesLock)
        {
            for (int i = 0; i < _choices.Count;)
            {
                if (_choices[i].At <= before)
                {
                    rows.Add(_choices[i].Row);
                    _choices.RemoveAt(i);
                }
                else
                {
                    i++;
                }
            }
        }
        return rows;
    }

    /// <summary>
    /// Rows recorded since the last call, oldest first. Empty outside combat,
    /// on the first call of a fresh combat's first decision, or on any error.
    /// Never throws.
    /// </summary>
    internal static List<Dictionary<string, object?>> Drain()
    {
        var rows = new List<Dictionary<string, object?>>();
        try
        {
            List<CombatHistoryEntry>? entries =
                CombatManager.Instance?.History?.Entries?.ToList();
            if (entries == null)
            {
                return rows;
            }
            if (entries.Count < _sent)
            {
                // A shorter history than last time can only be a new combat.
                Reset();
            }
            for (int i = _sent; i < entries.Count; i++)
            {
                rows.AddRange(TakeChoices(i));
                Dictionary<string, object?>? row = Map(entries[i]);
                if (row != null)
                {
                    rows.Add(row);
                }
            }
            _sent = entries.Count;
            rows.AddRange(TakeChoices(entries.Count));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCombatLog] Drain failed: {ex.Message}");
        }
        return rows;
    }

    /// <summary>One history entry as a wire row, or null if it maps to nothing.</summary>
    private static Dictionary<string, object?>? Map(CombatHistoryEntry entry)
    {
        Dictionary<string, object?> row = Header(entry);
        switch (entry)
        {
            case DamageReceivedEntry e:
                row["kind"] = "damage";
                row["dealer"] = Who(e.Dealer);
                row["receiver"] = Who(e.Receiver);
                row["unblocked"] = e.Result.UnblockedDamage + e.Result.OverkillDamage;
                row["blocked"] = e.Result.BlockedDamage;
                row["source"] = e.CardSource != null
                    ? CardSource(e.CardSource.Id.Entry, e.CardSource.IsUpgraded)
                    : OpenMoveSource();
                return row;

            case BlockGainedEntry e:
                row["kind"] = "block";
                row["receiver"] = Who(e.Actor);
                row["amount"] = e.Amount;
                row["source"] = e.CardPlay?.Card != null
                    ? CardSource(e.CardPlay.Card.Id.Entry, e.CardPlay.Card.IsUpgraded)
                    : OpenMoveSource();
                return row;

            case PowerReceivedEntry e:
                row["kind"] = "power";
                row["applier"] = Who(e.Applier);
                row["receiver"] = Who(e.Actor);
                row["power"] = e.Power?.Id.Entry;
                row["amount"] = (int)e.Amount;
                row["source"] = OpenMoveSource();
                return row;

            case MonsterPerformedMoveEntry e:
                row["kind"] = "monster_move";
                row["monster"] = Who(e.Actor);
                row["move_id"] = e.Move?.Id;
                // Open the move so the hits it deals can name it.
                _openMoveId = e.Move?.Id;
                _openMoveMonster = e.Monster?.Id.Entry;
                return row;

            case CardPlayStartedEntry e:
                row["kind"] = "card_play_started";
                row["card"] = e.CardPlay?.Card?.Id.Entry;
                row["upgraded"] = e.CardPlay?.Card?.IsUpgraded ?? false;
                row["auto_played"] = e.CardPlay?.IsAutoPlay ?? false;
                // Co-op: whose card and at what, so a recorded fight can be
                // replayed seat by seat. `play_index` > 0 is a repeat of the
                // same play (Echo Form, Burst), not a new one.
                row["owner"] = NetIdOf(() => e.CardPlay?.Card?.Owner?.NetId);
                row["target"] = Who(e.CardPlay?.Target);
                row["target_index"] = EnemyIndex(e.CardPlay?.Target);
                row["play_index"] = e.CardPlay?.PlayIndex ?? 0;
                return row;

            case CardPlayFinishedEntry e:
                row["kind"] = "card_play_finished";
                row["card"] = e.CardPlay?.Card?.Id.Entry;
                row["upgraded"] = e.CardPlay?.Card?.IsUpgraded ?? false;
                return row;

            case CardGeneratedEntry e:
                row["kind"] = "card_generated";
                row["card"] = e.Card?.Id.Entry;
                row["upgraded"] = e.Card?.IsUpgraded ?? false;
                row["by_player"] = e.Creator != null;
                return row;

            case CardDrawnEntry e:
                row["kind"] = "card_drawn";
                row["card"] = e.Card?.Id.Entry;
                row["upgraded"] = e.Card?.IsUpgraded ?? false;
                row["from_hand_draw"] = e.FromHandDraw;
                return row;

            case CardDiscardedEntry e:
                row["kind"] = "card_discarded";
                row["card"] = e.Card?.Id.Entry;
                row["upgraded"] = e.Card?.IsUpgraded ?? false;
                return row;

            case CardExhaustedEntry e:
                row["kind"] = "card_exhausted";
                row["card"] = e.Card?.Id.Entry;
                row["upgraded"] = e.Card?.IsUpgraded ?? false;
                return row;

            case OrbChanneledEntry e:
                row["kind"] = "orb_channeled";
                row["orb"] = e.Orb?.Id.Entry;
                return row;

            case PotionUsedEntry e:
                row["kind"] = "potion";
                row["potion"] = e.Potion?.Id.Entry;
                row["owner"] = NetIdOf(() => e.Potion?.Owner?.NetId);
                row["target_index"] = EnemyIndex(e.Target);
                return row;

            case EnergySpentEntry e:
                row["kind"] = "energy_spent";
                row["amount"] = e.Amount;
                return row;

            default:
                // An entry kind the workbench has no renderer for still reaches
                // the panel: `text` (below) is the game's own sentence, and the
                // Python side falls back to it for any unknown `kind`.
                row["kind"] = "other";
                return row;
        }
    }

    /// <summary>The fields every row carries: round, side and the game's own text.</summary>
    private static Dictionary<string, object?> Header(CombatHistoryEntry entry)
    {
        var row = new Dictionary<string, object?>
        {
            ["round"] = null,
            ["side"] = null,
            ["text"] = Safe(() => entry.Description),
            ["source"] = null,
        };
        string? human = Safe(() => entry.HumanReadableString);
        if (human != null)
        {
            Match m = HeaderRe.Match(human);
            if (m.Success)
            {
                if (int.TryParse(m.Groups[1].Value, NumberStyles.Integer,
                        CultureInfo.InvariantCulture, out int round))
                {
                    row["round"] = round;
                }
                // C# `CombatSide` is Player / Monster; the wire says enemy.
                row["side"] = m.Groups[2].Value == "Player" ? "player" : "enemy";
            }
        }
        return row;
    }

    /// <summary>A creature as `{kind, id, uid}`, matching the engine's shape.</summary>
    private static Dictionary<string, object?>? Who(Creature? creature)
    {
        if (creature == null)
        {
            return null;
        }
        return Safe(() => new Dictionary<string, object?>
        {
            ["kind"] = creature.IsPlayer ? "player" : "monster",
            ["id"] = creature.ModelId.Entry,
            ["uid"] = null,
            // Matches `combat_id` on the board's creatures (co-op replay).
            ["combat_id"] = creature.CombatId,
        });
    }

    /// <summary>A player's network id, as a string (a `ulong` does not
    /// survive every JSON reader). Matches `net_id` on the combat board.</summary>
    private static string? NetIdOf(Func<ulong?> netId)
    {
        try
        {
            return netId()?.ToString(CultureInfo.InvariantCulture);
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>The position of <paramref name="creature"/> among the combat's
    /// enemies, or null for a player, a pet or no target.</summary>
    private static int? EnemyIndex(Creature? creature)
    {
        if (creature == null)
        {
            return null;
        }
        try
        {
            int i = CombatManager.Instance?.DebugOnlyGetState()?.Enemies.ToList().IndexOf(creature) ?? -1;
            return i >= 0 ? i : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static Dictionary<string, object?> CardSource(string id, bool upgraded) =>
        new()
        {
            ["kind"] = "card",
            ["id"] = id,
            ["upgraded"] = upgraded,
        };

    /// <summary>The monster move currently mid-resolution, if any.</summary>
    private static Dictionary<string, object?>? OpenMoveSource()
    {
        if (_openMoveId == null)
        {
            return null;
        }
        return new Dictionary<string, object?>
        {
            ["kind"] = "move",
            ["monster"] = _openMoveMonster,
            ["move_id"] = _openMoveId,
        };
    }

    /// <summary>Run <paramref name="f"/>, swallowing anything it throws.</summary>
    private static T? Safe<T>(Func<T> f) where T : class
    {
        try
        {
            return f();
        }
        catch (Exception)
        {
            return null;
        }
    }
}
