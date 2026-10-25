// SimSpeed.cs -- live game-speed control (Engine.TimeScale).
//
// Mirrors the game's own debug shortcut NGame.DebugModifyTimescale
// (NGame.cs:1296-1302): round to one decimal, clamp to [0.1, 4.0], show the
// same fullscreen "TimeScale:x" VFX so the user gets feedback. Used by the
// settings slider, the speed hotkeys and the `spirementor speed` console
// command. Distinct from the Autopilot-only WaitSpeedPatch/
// AnimationSpeedPatch multipliers: TimeScale slows/speeds the WHOLE game
// uniformly and applies in any mode, which is exactly what "watch the AI at
// regular speed" and "play at 2x" want.
//
// Never applied automatically at startup -- only ever from an explicit user
// action (or this launch's own `--spire-mentor-speed`, SimMp) -- so a
// leftover value can never distort a sweep or surprise the player on launch.
//
// Engine.TimeScale is NOT ours alone: the game's hit-stop VFX
// (NHitStop.HitStopTask) drives it to 0.1 and ramps it back to a HARDCODED
// 1f, so a single hit-stop silently cancelled whatever speed the player or
// the config had chosen (reported 2026-08-08: 4.0x set in act 1 was 1.0x by
// act 2; CeremonialBeast/MechaKnight/Vantom are the v0.107.1 callers). So
// the chosen speed lives HERE, in _desired, and HitStopTimeScalePatch
// rewrites the hit-stop's writes as multipliers of it. Engine.TimeScale is
// the transient truth; _desired is the intent.

using System;
using System.Threading;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Vfx;

namespace SpireMentor;

public static class SimSpeed
{
    // The game's own clamp (NGame.DebugModifyTimescale).
    public const double Min = 0.1;
    public const double Max = 4.0;

    /// <summary>
    /// The speed that was asked for. Reads deliberately do NOT go to
    /// Engine.TimeScale: during a hit-stop that is a transient ramp value
    /// (0.1x upwards), and the settings slider / status lines must show what
    /// the player chose, not which frame of a screen-shake they read it on.
    /// </summary>
    public static double Desired => Volatile.Read(ref _desired);

    private static double _desired = 1.0;

    public static double Current => Desired;

    /// <summary>Set the game speed, clamped and rounded like the game's own
    /// debug shortcut. Safe from any thread (defers to the main thread).</summary>
    public static void Apply(double scale) => Apply(scale, showVfx: true);

    public static void Apply(double scale, bool showVfx)
    {
        double value = Math.Clamp(Math.Round(scale, 1), Min, Max);
        if (NGame.IsMainThread())
        {
            ApplyOnMainThread(value, showVfx);
        }
        else
        {
            Callable.From(() => ApplyOnMainThread(value, showVfx)).CallDeferred();
        }
    }

    /// <summary>Adjust the game speed by a delta (hotkeys / console). Steps
    /// from the DESIRED speed, not from Engine.TimeScale -- pressing F10
    /// during a hit-stop would otherwise step up from 0.1x.</summary>
    public static void Nudge(double offset) => Apply(Desired + offset);

    private static void ApplyOnMainThread(double value, bool showVfx)
    {
        try
        {
            bool changed = Math.Abs(Desired - value) > 1e-9;
            Volatile.Write(ref _desired, value);
            // Written unconditionally, even when the intent is unchanged: a
            // cancelled hit-stop can leave Engine.TimeScale part-way up its
            // ramp, and re-applying the same value is then a repair.
            Engine.TimeScale = value;
            if (changed && showVfx && NGame.Instance != null)
            {
                NFullscreenTextVfx? vfx = NFullscreenTextVfx.Create($"TimeScale:{Engine.TimeScale}");
                if (vfx != null)
                {
                    ((Node)NGame.Instance).AddChildSafely((Node?)vfx);
                }
            }
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[SpireMentor] SimSpeed.Apply failed: {ex.Message}");
        }
    }
}
