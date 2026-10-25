// SimTieBreak.cs -- say when the top fight moves are really a tie, and break
// such ties the way a player would: powers and self-buffs first.
//
// In a fight the search expects to win, every reasonable line wins and the
// options' Q values converge to within the search's own noise. The median
// gap between the best and the runner-up across 1,289 logged fight decisions
// was 0.0039 Q (~1 HP). The "best" is then a coin flip: one live case
// recommended Strike over Inflame+ by 0.0014 Q (0.4 HP), although Inflame
// first can only add damage to every Strike after it.
//
// So, client-side and display-only (the advice is a recommendation; nothing
// is played for the player):
//   * options within TieQ of the best and within TieWin of its win chance
//     are "about equal";
//   * among them, a Power card goes first, then a card that buffs the player
//     (a PowerVar of Strength / Dexterity / Vigor / Focus on a Self card),
//     then one that puts Vulnerable / Weak on enemies (Bash, Thunderclap,
//     Neutralize), then the search's own order, End turn last;
//   * the recommendation (star, card highlight, line, F5 peek) moves to that
//     card, and the HUD names the tie.
// Two copies of the same card tying is not a tie worth a note.

using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;

namespace SpireMentor;

internal static class SimTieBreak
{
    /// <summary>Q gap treated as noise: ~1.5 HP (1 HP ~ 0.0035 Q in logged fights).</summary>
    public const double TieQ = 0.005;

    /// <summary>Win-chance gap treated as noise.</summary>
    public const double TieWin = 0.01;

    /// <summary>Wider margin (~3 HP) when the pick is "End turn" while a
    /// harmless card is still playable -- e.g. End turn over a free Strike
    /// by 0.009 Q (96.7% vs 96.4% win) in a live Terror Eel fight.</summary>
    public const double EndTurnQ = 0.01;

    private static readonly string[] SelfBuffs =
        { "StrengthPower", "DexterityPower", "VigorPower", "FocusPower" };

    /// <summary>Debuffs that make the rest of the turn's hits count more
    /// (Bash, Thunderclap, Neutralize): first among tied attacks.</summary>
    private static readonly string[] EnemyDebuffs = { "VulnerablePower", "WeakPower" };

    /// <summary>Mark a near-tie on <paramref name="advice"/> and move the
    /// recommendation to the setup card in it, if any. Combat advice only.</summary>
    public static void Apply(SimAdvice advice, Player? me)
    {
        try
        {
            SimAdviceOption? best = advice.Options.Find(o => o.IsBest);
            if (me == null || best?.Q == null || advice.Best?.Action is not ("play" or "potion" or "end_turn"))
            {
                return;
            }
            IReadOnlyList<CardModel> hand = PileType.Hand.GetPile(me).Cards;
            bool endTurnBest = advice.Best.Action == "end_turn";
            double margin = endTurnBest ? EndTurnQ : TieQ;
            List<SimAdviceOption> tied = advice.Options
                .Where(o => o.Q != null && best.Q.Value - o.Q.Value <= margin &&
                            Math.Abs((o.PWin ?? 0) - (best.PWin ?? 0)) <= TieWin)
                .ToList();
            if (endTurnBest)
            {
                // Against "End turn" only cards that cannot hurt you: attacks,
                // powers and self-buffs -- never e.g. Bloodletting (costs HP),
                // which the replay showed the AI rightly holding back.
                tied = tied.Where(o => ReferenceEquals(o, best) || Harmless(o, hand)).ToList();
            }
            // Distinct moves: "Strike → A" and "Strike → B" are one card.
            List<string> names = tied.Select(o => CardName(o.Label)).Distinct().ToList();
            if (names.Count < 2)
            {
                return;
            }

            SimAdviceOption chosen = tied
                .Select((o, i) => (o, i, rank: Rank(o, hand)))
                .OrderBy(t => t.rank)
                .ThenBy(t => t.o.IsBest ? 0 : 1)
                .ThenByDescending(t => t.o.Q)
                .First().o;

            string tie = string.Join(" ≈ ", names.Take(3)) + (names.Count > 3 ? " ≈ …" : "");
            if (!ReferenceEquals(chosen, best) && chosen.CardIndex is int ci && ci >= 0 && ci < hand.Count)
            {
                best.IsBest = false;
                chosen.IsBest = true;
                advice.Best = new SimBestAction
                {
                    Action = "play",
                    CardIndex = ci,
                    TargetIndex = chosen.TargetIndex ?? -1,
                    Label = chosen.Label,
                };
                advice.Pv.Clear();
                advice.Pv.Add(chosen.Label);
                advice.TieNote = $"about equal: {tie} → {CardName(chosen.Label)} first (powers, buffs and debuffs before other moves)";
                Logger.Log($"[SimTieBreak] {tie}: recommending {chosen.Label} over {best.Label} " +
                           $"(Q {chosen.Q:F4} vs {best.Q:F4}).");
            }
            else if (!ReferenceEquals(chosen, best) && chosen.Label == "End turn" && best.CardIndex is null)
            {
                // A potion only tied with ending the turn: keep the potion.
                best.IsBest = false;
                chosen.IsBest = true;
                advice.Best = new SimBestAction { Action = "end_turn", Label = chosen.Label };
                advice.Pv.Clear();
                advice.Pv.Add(chosen.Label);
                advice.TieNote = $"about equal: {tie} → End turn (keep {best.Label} for a fight where it matters)";
                Logger.Log($"[SimTieBreak] {tie}: keeping {best.Label}, recommending End turn " +
                           $"(Q {chosen.Q:F4} vs {best.Q:F4}).");
            }
            else
            {
                advice.TieNote = $"about equal: {tie}";
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimTieBreak] skipped: {ex.Message}");
        }
    }

    /// <summary>An attack, power or self-buff that does not cost HP
    /// (HpLossVar: Hemokinesis, Bloodletting, Offering, ...).</summary>
    private static bool Harmless(SimAdviceOption o, IReadOnlyList<CardModel> hand) =>
        o.CardIndex is int ci && ci >= 0 && ci < hand.Count &&
        !hand[ci].DynamicVars.Values.Any(v => v is HpLossVar) &&
        (hand[ci].Type == CardType.Attack || Rank(o, hand) <= 2);

    /// <summary>0 = Power card, 1 = self-buff or hand setup (Armaments, Enlightenment, Rage), 2 = Vulnerable/Weak on enemies,
    /// 3 = anything else, 4 = End turn (when tied, do something rather than
    /// nothing), 5 = a potion: when drinking is only as good as the
    /// alternatives, keep it for a fight where it matters (the Dexterity
    /// Potions it recommended were all within 0.0006 of End turn).</summary>
    private static int Rank(SimAdviceOption o, IReadOnlyList<CardModel> hand)
    {
        if (o.Label == "End turn")
        {
            return 4;
        }
        if (o.CardIndex is not int ci || ci < 0 || ci >= hand.Count)
        {
            return 5; // potions
        }
        CardModel card = hand[ci];
        if (card.Type == CardType.Power)
        {
            return 0;
        }
        if (card.TargetType == TargetType.Self && HasPowerVar(card, SelfBuffs))
        {
            return 1;
        }
        // Hand setup: Armaments upgrades cards in hand, Enlightenment makes
        // them cheaper, Rage gives Block per Attack played this turn -- all
        // worth more before the cards they improve are played, so each counts
        // only while the hand holds such a card.
        string id = card.Id.Entry;
        if (id == "ENLIGHTENMENT" ||
            (id == "ARMAMENTS" && hand.Any(c => !ReferenceEquals(c, card) && c.IsUpgradable)) ||
            (id == "RAGE" && hand.Any(c => !ReferenceEquals(c, card) && c.Type == CardType.Attack)))
        {
            return 1;
        }
        if (card.TargetType is TargetType.AnyEnemy or TargetType.AllEnemies or TargetType.RandomEnemy &&
            HasPowerVar(card, EnemyDebuffs))
        {
            return 2;
        }
        return 3;
    }

    private static bool HasPowerVar(CardModel card, string[] powers) =>
        card.DynamicVars.Values.Any(v =>
        {
            Type t = v.GetType();
            return t.IsGenericType && t.GetGenericTypeDefinition() == typeof(PowerVar<>) &&
                   powers.Contains(t.GetGenericArguments()[0].Name);
        });

    /// <summary>"Strike → Corpse Slug" -> "Strike".</summary>
    private static string CardName(string label)
    {
        int arrow = label.IndexOf(" → ", StringComparison.Ordinal);
        return (arrow >= 0 ? label[..arrow] : label).Trim();
    }
}
