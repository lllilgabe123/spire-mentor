// SimAdvisorAutoplay.cs -- hands-free coached run (`advisor_autoplay` in
// SpireMentor.conf).
//
// Why this exists: the advisor lane is reachable ONLY through the keyboard
// (F8 to enter advisor mode, F7 to arm autoplay), so nothing about it could
// be regression-tested without a human sitting at the game. That is how six
// F7 stalls shipped at once on 2026-08-06 -- every one of them reproduces in
// the first ten minutes of an unattended run, and none of them could be
// reproduced by any harness we had.
//
// What it does, once and only once per launch: wait until a run is actually
// live, then take the same two steps a player takes -- hand the game back
// from autopilot (F8's `MainFile.StopAutopilot` + Mode = Advisor) and arm
// autoplay (F7's `SimAdvisor.ToggleAutoPlay`). Deliberately the SAME entry
// points as the hotkeys: a harness that called some private back door would
// stop testing the thing players use.
//
// Getting INTO a run is left to `auto_start=true`: SimAutoSlayer's
// PlayMainMenuAsync already drives character select and Embark, and with no
// bridge client connected it then blocks in SimAgentGate at the first
// decision without having touched the run (SimRuntime.AutopilotEngaged is
// false, so every behaviour patch stays inert). That block is exactly where
// the advisor lane takes over.
//
// Default OFF, and never persisted by the settings page -- the sweep lane
// reads the same file, and a stray `advisor_autoplay` would quietly convert
// a parity sweep into an advisor session.

using System;
using Godot;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

internal static class SimAdvisorAutoplay
{
    /// <summary>How often to check whether a run has started.</summary>
    private const double PollSeconds = 1.0;

    /// <summary>
    /// Beat between entering advisor mode and arming autoplay. The mode flip
    /// launches the Python advisor and installs the observation hooks; arming
    /// before it has a socket just means the first decision is answered a
    /// little later, but this keeps the log readable.
    /// </summary>
    private const double ArmDelaySeconds = 3.0;

    private static bool _fired;

    public static void ArmWhenRunStarts()
    {
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                Logger.Log("[SimAdvisorAutoplay] No scene tree; hands-free autoplay not armed.");
                return;
            }
            var timer = new Godot.Timer
            {
                Name = "SpireMentorAdvisorAutoplayArm",
                WaitTime = PollSeconds,
                Autostart = true,
                OneShot = false,
            };
            timer.Connect(Godot.Timer.SignalName.Timeout, Callable.From(() => Poll(timer)));
            tree.Root.AddChildSafely(timer);
            Logger.Log("[SimAdvisorAutoplay] Armed: will hand the run to the advisor lane once it starts.");
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisorAutoplay] Could not arm: {ex.Message}");
        }
    }

    private static void Poll(Godot.Timer timer)
    {
        if (_fired)
        {
            return;
        }
        try
        {
            RunState? runState = RunManager.Instance.DebugOnlyGetState();
            // A state alone is not enough -- it exists before the first room
            // does, and taking over then would leave the advisor describing a
            // half-built run. Same condition SimAutoSlayer waits on.
            if (runState?.CurrentRoom == null ||
                runState.CurrentRoom.RoomType == RoomType.Unassigned)
            {
                return;
            }
            // `debug_player_hp` is applied by SimAutoSlayer at the top of its
            // room loop, which is also the first moment the room type is
            // assigned -- so handing over on room-assignment alone is a race
            // that silently loses the cheat, and a 999-HP screen-coverage run
            // quietly becomes a 70-HP one that dies in act 1. Wait for it.
            int wantHp = SimConfig.Instance.DebugPlayerHp;
            if (wantHp > 0)
            {
                var creature = LocalContext.GetMe(runState)?.Creature;
                if (creature == null || creature.MaxHp < wantHp)
                {
                    return;
                }
            }
            _fired = true;
            if (GodotObject.IsInstanceValid(timer))
            {
                timer.Stop();
                timer.QueueFreeSafely();
            }

            Logger.Log("[SimAdvisorAutoplay] Run is live; switching to advisor mode.");
            MainFile.StopAutopilot();
            SimRuntime.Mode = SimMode.Advisor;
            SimActionLog.Note("hands-free: advisor mode on (advisor_autoplay)");

            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return;
            }
            SceneTreeTimer arm = tree.CreateTimer(ArmDelaySeconds);
            arm.Connect(SceneTreeTimer.SignalName.Timeout, Callable.From(() =>
            {
                if (SimRuntime.Mode != SimMode.Advisor)
                {
                    return;
                }
                Logger.Log("[SimAdvisorAutoplay] Arming autoplay (same path as Ctrl+Shift+P).");
                SimAdvisor.Instance.ToggleAutoPlay();
            }));
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimAdvisorAutoplay] Handover failed: {ex.Message}");
        }
    }
}
