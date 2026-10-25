// SimActions.cs -- the mod's user-facing verbs, shared by every surface.
//
// Hotkeys (SimHotkeys), the toolbar (SimToolbar), the dev-console command
// (SimConsoleCommands) and the BaseLib settings page all funnel through
// here, so behavior (and the log lines users see) cannot drift between
// surfaces. Bodies moved verbatim from SimHotkeys (2026-08-21).

using System;
using Godot;

namespace SpireMentor;

internal static class SimActions
{
    // One press, one action. With some mod sets every click / key press
    // reaches the toolbar and the hotkeys TWICE (observed live: one click on
    // the style button logged "Style -> Hints" then "Style -> Coach", one F5
    // two peeks), so every toggle landed back where it started. A repeat of
    // the same verb within this window is that duplicate, not a human.
    private const long RepeatWindowMs = 250;
    private static readonly Dictionary<string, long> _lastRun = new();

    private static bool IsRepeat(string verb)
    {
        long now = System.Environment.TickCount64;
        lock (_lastRun)
        {
            if (_lastRun.TryGetValue(verb, out long last) && now - last < RepeatWindowMs)
            {
                GD.Print($"[SpireMentor] Ignored a repeated '{verb}' {now - last} ms after the first.");
                return true;
            }
            _lastRun[verb] = now;
            return false;
        }
    }

    public static void ToggleHelp()
    {
        if (!IsRepeat("help")) SimHelp.Toggle();
    }

    public static void ToggleLog()
    {
        if (!IsRepeat("log")) SimActionLog.TogglePanel();
    }

    public static void NudgeSpeed(double offset)
    {
        if (!IsRepeat(offset < 0 ? "speed-" : "speed+")) SimSpeed.Nudge(offset);
    }

    /// <summary>Copy this position to the clipboard for the workbench's
    /// live-state import (SimStateCopy).</summary>
    public static void CopyState()
    {
        if (!IsRepeat("copy")) SimStateCopy.Copy();
    }

    /// <summary>Undo to the start of the current combat turn (replays the
    /// game's own combat tape -- SimCombatUndo).</summary>
    public static void UndoTurn()
    {
        if (!IsRepeat("undo")) SimCombatUndo.RequestUndoTurn();
    }

    /// <summary>Rewind to the start of the current room (same seed, exact
    /// state -- the game's own room-entry save).</summary>
    public static void RestartRoom()
    {
        if (IsRepeat("restart-room"))
        {
            return;
        }
        SimSnapshotMeta? target = SimSnapshots.CurrentRoomEntry();
        if (target == null)
        {
            SimActionLog.Note("Nothing to rewind to yet (the game saves at each room entry).");
            return;
        }
        SimRestore.RequestRestore(target.Id);
    }

    /// <summary>Rewind one room further back.</summary>
    public static void BackOneRoom()
    {
        if (IsRepeat("back-room"))
        {
            return;
        }
        SimSnapshotMeta? target = SimSnapshots.PreviousRoomEntry();
        if (target == null)
        {
            SimActionLog.Note("No earlier room snapshot in this run yet.");
            return;
        }
        SimRestore.RequestRestore(target.Id);
    }

    public static void ToggleHud()
    {
        if (!IsRepeat("hud")) SetHud(!SpireMentorModConfig.ShowHud);
    }

    public static void SetHud(bool on)
    {
        SpireMentorModConfig.ShowHud = on;
        GD.Print($"[SpireMentor] HUD {(on ? "shown" : "hidden")}");
        SimRuntime.NotifyChanged();
    }

    /// <summary>AI takes this one decision (F6 / toolbar).</summary>
    public static void AiOnce(string surface)
    {
        if (!IsRepeat("ai-once")) RequireAdvisor(surface, () => SimAdvisor.Instance.ExecuteBestOnce());
    }

    /// <summary>AI plays until stopped (F7 / toolbar), toggled.</summary>
    public static void ToggleAutoplay(string surface)
    {
        if (!IsRepeat("ai-auto")) RequireAdvisor(surface, () => SimAdvisor.Instance.ToggleAutoPlay());
    }

    /// <summary>
    /// F8 / the on-off button: AI off &lt;-&gt; on in the current style
    /// (CoachStyle.ToggleOnOff). From Autopilot: stop the AI, hand the game
    /// back, and land in the current style. (Autopilot can only be entered at
    /// launch, from the config file.)
    /// </summary>
    public static void ToggleAdvisor()
    {
        if (IsRepeat("ai-on-off"))
        {
            return;
        }
        (SimModeLite nextMode, AdviceStyle nextStyle) =
            CoachStyle.ToggleOnOff((SimModeLite)(int)SimRuntime.Mode, SimRuntime.Style);
        if (SimRuntime.Mode == SimMode.Autopilot)
        {
            // Taking over from the AI: stop the slayer FIRST so its card
            // selector and non-interactive mode are released before the mode
            // flips (see MainFile.StopAutopilot).
            MainFile.StopAutopilot();
            SimRuntime.SetModeAndStyle((SimMode)(int)nextMode, nextStyle);
            string style = CoachStyle.StyleLabel(nextStyle);
            GD.Print($"[SpireMentor] Autopilot stopped -> {style} (you have the controls).");
            SimActionLog.Note($"Autopilot stopped; you have the controls ({style}).");
            return;
        }
        SimRuntime.SetModeAndStyle((SimMode)(int)nextMode, nextStyle);
        string name = CoachStyle.ModeName(nextMode, nextStyle);
        GD.Print($"[SpireMentor] Mode -> {name}");
        if (nextMode == SimModeLite.Advisor)
        {
            SimActionLog.Note(nextStyle == AdviceStyle.Coach
                ? "Coach: nothing shows before you act -- F5 peeks at the AI's next pick."
                : "Hints: the AI's opinion shows while you decide.");
        }
    }

    /// <summary>
    /// The style button: Hints &lt;-&gt; Coach (CoachStyle.SwitchStyle). Never
    /// changes the mode, so it can't turn the AI (or its process) off.
    /// </summary>
    public static void SwitchStyle()
    {
        if (IsRepeat("style"))
        {
            return;
        }
        (SimModeLite mode, AdviceStyle nextStyle) =
            CoachStyle.SwitchStyle((SimModeLite)(int)SimRuntime.Mode, SimRuntime.Style);
        SimRuntime.SetModeAndStyle((SimMode)(int)mode, nextStyle);
        GD.Print($"[SpireMentor] Style -> {CoachStyle.StyleLabel(nextStyle)}");
        SimActionLog.Note(nextStyle == AdviceStyle.Coach
            ? "Coach: nothing shows before you act -- F5 peeks at the AI's next pick."
            : "Hints: the AI's opinion shows while you decide.");
    }

    /// <summary>
    /// F5: in the Coach style, show the AI's pick for just the next decision
    /// (SimAdvisor.Peek). Hints already shows everything, so F5 does nothing
    /// there; with the coach off or in Autopilot it only says what it is for
    /// (the RequireAdvisor pattern).
    /// </summary>
    public static void Peek()
    {
        if (IsRepeat("peek"))
        {
            return;
        }
        GD.Print($"[SpireMentor] F5 pressed (mode {SimRuntime.Mode}, style {SimRuntime.Style}).");
        if (SimRuntime.Mode == SimMode.Advisor && SimRuntime.Style == AdviceStyle.Hints)
        {
            SimToolbar.Instance.SetNote("Hints already shows the AI's pick (F5 peeks in Coach)");
            return;
        }
        if (!SimRuntime.CoachStyleActive)
        {
            GD.Print("[SpireMentor] F5 peeks in Coach mode.");
            SimToolbar.Instance.SetNote("F5 peeks in Coach mode");
            SimActionLog.Note("F5 peeks in Coach mode (Insert turns the AI on; the toolbar's style button picks Hints or Coach).");
            return;
        }
        SimAdvisor.Instance.Peek();
    }

    /// <summary>
    /// The takeover verbs only mean anything in Advisor mode; say so instead
    /// of looking broken (the advisor's own guards would silently no-op).
    /// <paramref name="what"/> names the surface the user pressed, e.g.
    /// "Ctrl+Shift+N (AI takes this decision)" or "the AI-once button".
    /// </summary>
    public static void RequireAdvisor(string what, Action action)
    {
        if (SimRuntime.Mode != SimMode.Advisor)
        {
            string how = SimRuntime.Mode == SimMode.Autopilot
                ? "the AI is already playing (Autopilot) -- press Insert to take over"
                : "press Insert to turn Advisor mode on";
            GD.Print($"[SpireMentor] {what} needs Advisor mode: {how}.");
            SimActionLog.Note($"{what} needs Advisor mode: {how}.");
            return;
        }
        action();
    }
}
