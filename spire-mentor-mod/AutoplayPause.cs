// AutoplayPause.cs -- when F7 autoplay stops and hands a decision back to the
// player (`pause_on` in SpireMentor.conf, GH #874).
//
// F7 used to play every screen it reached, including the ones it has no
// policy for: the advisor's unranked first-option pick, a borrowed head's
// ranking, the crystal sphere's blind reveals. Nothing told the player it was
// happening. `pause_on` lists what F7 must stop on instead:
//
//   "unsupported"  any decision the advisor flags with `advice.hand_back`
//                  (or the mod flags for a screen it answers itself);
//   any other id   an event id ("Neow", "CrystalSphere"), an `advice_for`
//                  ("shop", "card_bundle") or a screen name -- matched
//                  case- and punctuation-blind, so "Neow" finds NEOW and
//                  "crystal sphere" finds CRYSTAL_SPHERE.
//
// "Unsupported" is decided by the advisor, not by a list here: when a surface
// gains a trained head its answers stop carrying `hand_back`, and the pause
// goes away with no mod change.
//
// Game-free on purpose (no Godot, no sts2.dll) so tests/unit can link it.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace SpireMentor;

internal static class AutoplayPause
{
    public const string Unsupported = "unsupported";

    /// <summary>What a conf without `pause_on` gets: a player install pauses
    /// on everything the AI cannot actually decide.</summary>
    public static readonly IReadOnlyList<string> Default = new[] { Unsupported };

    /// <summary>
    /// `pause_on` as written: a comma-separated string ("unsupported, Neow"),
    /// or a JSON array of strings. An empty string, empty array or null means
    /// never pause (unattended runs). Absent or any other shape: the fallback.
    /// </summary>
    public static IReadOnlyList<string> FromJson(JsonElement root, string name, IReadOnlyList<string> fallback)
    {
        if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty(name, out JsonElement el))
        {
            return fallback;
        }
        return el.ValueKind switch
        {
            JsonValueKind.String => Parse(el.GetString()),
            JsonValueKind.Null => Array.Empty<string>(),
            JsonValueKind.Array => Parse(string.Join(",", el.EnumerateArray()
                .Where(e => e.ValueKind == JsonValueKind.String)
                .Select(e => e.GetString()))),
            _ => fallback,
        };
    }

    /// <summary>Split a comma-separated list into normalized, distinct ids.</summary>
    public static IReadOnlyList<string> Parse(string? raw) =>
        (raw ?? "").Split(',')
            .Select(Normalize)
            .Where(id => id.Length > 0)
            .Distinct()
            .ToList();

    /// <summary>Lower-case letters and digits only: "Crystal_Sphere",
    /// "CRYSTAL_SPHERE" and "crystal sphere" are one id.</summary>
    public static string Normalize(string? id)
    {
        var sb = new StringBuilder((id ?? "").Length);
        foreach (char c in id ?? "")
        {
            if (char.IsLetterOrDigit(c))
            {
                sb.Append(char.ToLowerInvariant(c));
            }
        }
        return sb.ToString();
    }

    /// <summary>
    /// The list that actually applies. `advisor_autoplay` is the unattended
    /// regression lane: nobody is there to take a handed-back screen, so it
    /// never pauses whatever the conf says.
    /// </summary>
    public static IReadOnlyList<string> Effective(IReadOnlyList<string> pauseOn, bool advisorAutoplay) =>
        advisorAutoplay ? Array.Empty<string>() : pauseOn;

    /// <summary>
    /// Why autoplay should stop on this decision, or null to play on.
    /// <paramref name="handBack"/> is the advisor's (or the mod's) verdict
    /// that it cannot really decide this; <paramref name="ids"/> are the
    /// names the decision answers to (advice_for, screen, event id).
    /// </summary>
    public static string? Reason(IReadOnlyList<string> pauseOn, string? handBack, IEnumerable<string?> ids)
    {
        if (pauseOn.Count == 0)
        {
            return null;
        }
        if (!string.IsNullOrWhiteSpace(handBack) && pauseOn.Contains(Unsupported))
        {
            return $"not covered by the AI ({handBack.Trim()})";
        }
        string? hit = ids.Select(Normalize).FirstOrDefault(id => id.Length > 0 && id != Unsupported && pauseOn.Contains(id));
        return hit == null ? null : "you asked to play this yourself (pause_on)";
    }

    /// <summary>"CRYSTAL_SPHERE" / "card_bundle" -> "Crystal Sphere" / "Card Bundle".</summary>
    public static string Pretty(string? id)
    {
        string[] words = (id ?? "").Split(new[] { '_', ' ', '-' }, StringSplitOptions.RemoveEmptyEntries);
        return words.Length == 0
            ? "This decision"
            : string.Join(" ", words.Select(w => char.ToUpperInvariant(w[0]) + w.Substring(1).ToLowerInvariant()));
    }

    /// <summary>The line the player sees when autoplay hands back.</summary>
    public static string Toast(string what, string reason) =>
        $"{what}: {reason} — your call. Ctrl+Shift+P to resume auto mode.";
}
