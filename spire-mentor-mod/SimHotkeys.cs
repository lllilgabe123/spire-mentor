// SimHotkeys.cs -- global hotkeys for live mod control.
//
// Uses the game's own NHotkeyManager (NHotkeyManager.cs): bindings are
// looked up by InputMap ACTION NAME, invoked via CallDeferred (so handlers
// run on the main thread), skipped while the dev console is open or a
// LineEdit has focus, and SetInputAsHandled() only fires when a binding
// matched. Our custom action names are not in MegaInput.AllInputs, so the
// game's "blocking screen" mechanism (pause menu etc.) never blocks them --
// which is what we want: mode/speed/HUD control should work on any screen.
//
// Defaults (chosen from keys the game leaves unbound):
//   Home          help panel -- mode, every key, where the files are
//   F4            toggle HUD visibility
//   F3            toggle the action-log panel
//   F5            Coach style: peek at the AI's pick for the next decision only
//   Ctrl+Shift+N  AI takes this decision (advisor mode, combat)
//   Ctrl+Shift+P  AI plays until I stop (toggle)
//   Insert        AI on / off in the current style (from Autopilot: stop the AI and take over)
//   F9  game speed down 0.1x
//   F10 game speed up 0.1x
//
// Key survey (v0.107.1 decomp), so these don't fight the game:
//   F2  is TAKEN and ungated -- NFeedbackScreenOpener.cs:72 opens the
//       feedback screen on keycode 4194333 with no debug/release check. It
//       was our HUD key until 2026-08-05, when pressing it in a live session
//       opened feedback instead. Do not re-use it.
//   F11 is TAKEN -- NDevConsole.
//   F1/F3-F10 map to DebugHotkey.* actions in NInputManager's _debugInputMap,
//       but ProcessDebugKeyInput returns early unless NGame.IsTrailerMode
//       (NInputManager.cs:463), which only the `trailer` console command
//       sets -- so they are free in a normal session. F1 (keycode 4194332 =
//       DebugHotkey.hideRestSite) was re-checked for the help panel, 0.2.1.

using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace SpireMentor;

public static class SimHotkeys
{
    private const string ActionHelp = "spirementor_help";
    private const string ActionToggleHud = "spirementor_toggle_hud";
    private const string ActionToggleLog = "spirementor_toggle_log";
    private const string ActionPeek = "spirementor_peek";
    private const string ActionAiOnce = "spirementor_ai_once";
    private const string ActionAiAuto = "spirementor_ai_auto";
    private const string ActionToggleAdvisor = "spirementor_toggle_advisor";
    private const string ActionSpeedDown = "spirementor_speed_down";
    private const string ActionSpeedUp = "spirementor_speed_up";
    private const string ActionUndoTurn = "spirementor_undo_turn";
    private const string ActionRestartRoom = "spirementor_restart_room";
    private const string ActionCopyState = "spirementor_copy_state";

    private static bool _installed;

    /// <summary>Wait for the game to come up, then install on the main
    /// thread. Run via TaskHelper.RunSafely from MainFile.Initialize.</summary>
    public static async Task InstallWhenReadyAsync()
    {
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(120));
        await WaitHelper.Until(
            () => NGame.Instance != null && NHotkeyManager.Instance != null,
            cts.Token, TimeSpan.FromSeconds(120), "NHotkeyManager not available");
        Callable.From(Install).CallDeferred();
    }

    private static void Install()
    {
        if (_installed)
        {
            return;
        }
        try
        {
            NHotkeyManager? manager = NHotkeyManager.Instance;
            if (manager == null)
            {
                return;
            }

            // Help, AI once / auto and AI on/off moved off F1/F6/F7/F8: other
            // installed mods (BetterSpire2, Damage Meter, Wait for Vuln, Low
            // Spec Spire, Who Carried?, Map Enhance, Host Tools) listen to
            // those. Home and Insert are bound by neither the game nor them;
            // N and P are not game letters, so the Ctrl+Shift chords cannot
            // also fire a game action (NHotkeyManager uses IsActionPressed,
            // which ignores extra modifiers).
            RegisterAction(ActionHelp, Key.Home);
            RegisterAction(ActionToggleHud, Key.F4);
            RegisterAction(ActionToggleLog, Key.F3);
            RegisterAction(ActionPeek, Key.F5);
            RegisterAction(ActionAiOnce, Key.N, ctrl: true, shift: true);
            RegisterAction(ActionAiAuto, Key.P, ctrl: true, shift: true);
            RegisterAction(ActionToggleAdvisor, Key.Insert);
            RegisterAction(ActionSpeedDown, Key.F9);
            RegisterAction(ActionSpeedUp, Key.F10);
            // Ctrl-chorded, not F-keys: Z alone is a card hotkey surface and
            // the remaining F-keys are taken (F2 feedback, F11 console).
            RegisterAction(ActionUndoTurn, Key.Z, ctrl: true);
            RegisterAction(ActionRestartRoom, Key.Z, ctrl: true, shift: true);
            // Also Ctrl-chorded, and SHIFTed: plain Ctrl+C is the OS copy every
            // text field expects, and stealing it from the settings panel's
            // fields would be a worse trade than a two-modifier chord.
            RegisterAction(ActionCopyState, Key.C, ctrl: true, shift: true);

            manager.PushHotkeyPressedBinding(ActionHelp, SimActions.ToggleHelp);
            manager.PushHotkeyPressedBinding(ActionToggleHud, SimActions.ToggleHud);
            manager.PushHotkeyPressedBinding(ActionToggleLog, SimActions.ToggleLog);
            manager.PushHotkeyPressedBinding(ActionPeek, SimActions.Peek);
            manager.PushHotkeyPressedBinding(ActionAiOnce, () => SimActions.AiOnce("Ctrl+Shift+N (AI takes this decision)"));
            manager.PushHotkeyPressedBinding(ActionAiAuto, () => SimActions.ToggleAutoplay("Ctrl+Shift+P (AI plays until you stop)"));
            manager.PushHotkeyPressedBinding(ActionToggleAdvisor, SimActions.ToggleAdvisor);
            manager.PushHotkeyPressedBinding(ActionSpeedDown, () => SimActions.NudgeSpeed(-0.1));
            manager.PushHotkeyPressedBinding(ActionSpeedUp, () => SimActions.NudgeSpeed(0.1));
            manager.PushHotkeyPressedBinding(ActionUndoTurn, SimActions.UndoTurn);
            manager.PushHotkeyPressedBinding(ActionRestartRoom, SimActions.RestartRoom);
            manager.PushHotkeyPressedBinding(ActionCopyState, SimActions.CopyState);

            _installed = true;
            GD.Print("[SpireMentor] Hotkeys installed: Home=help, F4=HUD, F3=log, F5=peek (Coach), Ctrl+Shift+N=AI once, Ctrl+Shift+P=AI auto, Insert=AI on/off, F9/F10=speed -/+, Ctrl+Z=undo turn, Ctrl+Shift+Z=restart room, Ctrl+Shift+C=copy state");
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[SpireMentor] Hotkey install failed: {ex.Message}");
        }
    }

    private static void RegisterAction(string name, Key key, bool ctrl = false, bool shift = false)
    {
        var actionName = new StringName(name);
        if (InputMap.HasAction(actionName))
        {
            return;
        }
        InputMap.AddAction(actionName);
        var ev = new InputEventKey { Keycode = key, CtrlPressed = ctrl, ShiftPressed = shift };
        InputMap.ActionAddEvent(actionName, ev);
    }

}
