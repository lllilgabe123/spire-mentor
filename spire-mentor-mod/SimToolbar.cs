// SimToolbar.cs -- the clickable mod strip (discoverability layer).
//
// Everything the F-keys do, as buttons: mode toggle, AI once/auto, speed,
// then three dropdowns -- Rewind, Panels, Workbench -- so nobody has to
// memorise a key list to use the mod. The live controls (coach state, AI
// once/auto, speed) stay flat because their state is read at a glance;
// the one-shot verbs are grouped by job so the strip fits a 1280-wide
// window. Hotkeys stay as shortcuts; every entry routes through the same
// SimActions verbs.
//
// Input-safety contract (see SimUiKit): only the buttons are
// MouseFilter.Stop; every container, gap and label is Ignore, so clicks
// anywhere but a visible button land in the game. Collapse is CLICK-driven
// via an always-visible pill -- hover-collapse would put a moving Stop
// region over combat. The strip sits at CanvasLayer 98 (below HUD 99 / log
// 100 / help 101) at the top-left, pushed below the game's top bar by
// SimTopBarLayout.
//
// Refresh model: SimRuntime.Changed handles mode flips from any surface;
// a 150 ms Godot.Timer handles the state that raises no event (SimSpeed,
// AdvisorAutoplayArmed, top-bar drift). No source generators -> no _Process
// (see SpireMentor.csproj header).

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes;

namespace SpireMentor;

internal sealed class SimToolbar
{
    public static readonly SimToolbar Instance = new();

    private SimToolbar()
    {
        SimRuntime.Changed += () => RunOnMainThread(Refresh);
    }

    private CanvasLayer? _layer;
    private VBoxContainer? _container;
    private Button? _pill;
    private PanelContainer? _strip;
    private HBoxContainer? _buttons;
    private Button? _coach;
    private Button? _style;
    private Button? _peek;
    private Button? _aiOnce;
    private Button? _aiAuto;
    private Label? _speedLabel;
    private Label? _statusLabel;
    private Label? _hintLabel;
    private readonly Dictionary<string, Button> _extensions = new();

    private bool _expanded = true;
    private SimMode _lastMode = SimMode.Off;
    private bool _modeSeen;

    private SimAdvisorStatus _advisorStatus = SimAdvisorStatus.None;
    /// <summary>When the current engine build started, for the pill's counter.</summary>
    private DateTime? _buildStartedUtc;
    private string _lastActionSummary = "";
    private string? _note;

    // ------------------------------------------------------------------
    // Public entry points (any thread)
    // ------------------------------------------------------------------

    /// <summary>Wait for the game to come up, then build on the main thread.
    /// Run via TaskHelper.RunSafely from MainFile.Initialize.</summary>
    public async Task InstallWhenReadyAsync()
    {
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(120));
        await WaitHelper.Until(
            () => NGame.Instance != null,
            cts.Token, TimeSpan.FromSeconds(120), "NGame not available for SimToolbar");
        RunOnMainThread(() =>
        {
            if (EnsureBuilt())
            {
                Refresh();
            }
        });
    }

    public void SetAdvisorStatus(SimAdvisorStatus status)
    {
        RunOnMainThread(() =>
        {
            // Stamp the start so the pill can count the wait up. A build this
            // long has to prove it is still alive; a static label does not.
            if (status == SimAdvisorStatus.Building && _advisorStatus != SimAdvisorStatus.Building)
            {
                _buildStartedUtc = DateTime.UtcNow;
            }
            else if (status != SimAdvisorStatus.Building)
            {
                _buildStartedUtc = null;
            }
            _advisorStatus = status;
            Refresh();
        });
    }

    public void SetLastAction(string summary)
    {
        RunOnMainThread(() =>
        {
            _lastActionSummary = summary;
            Refresh();
        });
    }

    public void SetNote(string? note)
    {
        RunOnMainThread(() =>
        {
            _note = note;
            Refresh();
        });
    }

    /// <summary>
    /// Extension point for later features (e.g. the rewind cluster): appends
    /// a button to the strip, once per id. Safe from any thread.
    /// </summary>
    public void AddExtensionButton(string id, string label, string hint, Action onPressed)
    {
        RunOnMainThread(() =>
        {
            if (!EnsureBuilt() || _buttons == null || _extensions.ContainsKey(id))
            {
                return;
            }
            Button button = SimUiKit.MakeButton(label, hint, onPressed, SetHint);
            _buttons.AddChild(button);
            _extensions[id] = button;
        });
    }

    /// <summary>Bottom edge (global Y) of the visible strip, or null when not
    /// built -- SimHud parks the advice panel below this.</summary>
    public float? BottomY()
    {
        if (_container == null || !GodotObject.IsInstanceValid(_container) ||
            !_container.IsVisibleInTree())
        {
            return null;
        }
        Rect2 rect = _container.GetGlobalRect();
        return rect.Position.Y + rect.Size.Y;
    }

    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    private bool EnsureBuilt()
    {
        if (_layer != null && GodotObject.IsInstanceValid(_layer))
        {
            return true;
        }
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return false;
            }

            _expanded = SpireMentorModConfig.ToolbarExpanded && SimRuntime.Mode != SimMode.Off;

            var layer = new CanvasLayer
            {
                Name = "SpireMentorToolbarLayer",
                Layer = 98,
            };

            var container = new VBoxContainer
            {
                Name = "SpireMentorToolbar",
                Position = new Vector2(PanelLeft, SimTopBarLayout.FallbackTop),
                MouseFilter = Control.MouseFilterEnum.Ignore,
            };
            container.AddThemeConstantOverride("separation", 2);

            // Row 1: the pill + the button strip.
            var row = new HBoxContainer
            {
                Name = "ToolbarRow",
                MouseFilter = Control.MouseFilterEnum.Ignore,
            };
            row.AddThemeConstantOverride("separation", 6);

            _pill = SimUiKit.MakeButton("Spire Mentor", "Show / hide the Spire Mentor toolbar", TogglePill, SetHint);
            row.AddChild(_pill);

            var strip = new PanelContainer
            {
                Name = "ToolbarStrip",
                MouseFilter = Control.MouseFilterEnum.Ignore,
                Visible = false,
            };
            strip.AddThemeStyleboxOverride("panel", SimUiKit.PanelStyle());

            var buttons = new HBoxContainer
            {
                Name = "ToolbarButtons",
                MouseFilter = Control.MouseFilterEnum.Ignore,
            };
            buttons.AddThemeConstantOverride("separation", 4);

            _coach = SimUiKit.MakeButton("AI off",
                "Turn the AI coach on / off (Insert)",
                SimActions.ToggleAdvisor, SetHint);
            buttons.AddChild(_coach);

            _style = SimUiKit.MakeButton("Coach",
                "Switch style: Hints (the AI's opinion while you decide) <-> Coach (nothing " +
                "until you act; F5 peeks, a note afterwards says where you differed)",
                SimActions.SwitchStyle, SetHint);
            buttons.AddChild(_style);

            // Only shown in the Coach style (Hints already shows the pick).
            _peek = SimUiKit.MakeButton("Peek",
                "Show the AI's pick for just this decision; it hides again once you act (F5)",
                SimActions.Peek, SetHint);
            buttons.AddChild(_peek);

            _aiOnce = SimUiKit.MakeButton("AI once",
                "Let the AI take just this one decision (Ctrl+Shift+N) -- needs the coach on",
                () => SimActions.AiOnce("The AI-once button"), SetHint);
            buttons.AddChild(_aiOnce);

            _aiAuto = SimUiKit.MakeButton("AI auto",
                "Let the AI keep playing until you stop it (Ctrl+Shift+P) -- needs the coach on",
                () => SimActions.ToggleAutoplay("The AI-auto button"), SetHint);
            buttons.AddChild(_aiAuto);

            buttons.AddChild(SimUiKit.MakeButton("-", "Game speed down 0.1x (F9)",
                () => SimActions.NudgeSpeed(-0.1), SetHint));
            _speedLabel = SimUiKit.SmallLabel("1.0x", Colors.White);
            _speedLabel.VerticalAlignment = VerticalAlignment.Center;
            buttons.AddChild(_speedLabel);
            buttons.AddChild(SimUiKit.MakeButton("+", "Game speed up 0.1x (F10)",
                () => SimActions.NudgeSpeed(0.1), SetHint));

            // Everything past the live controls is a one-shot verb, grouped
            // by job into three dropdowns (same shape as the workbench's top
            // bar): 16 flat buttons used to run off a 1280-wide window.
            //
            // Rewind cluster: the game saves at room entry and post-combat,
            // so those are the room-level rewind targets (see SimSnapshots);
            // the turn undo replays the game's own combat tape on top of the
            // room-entry save (SimCombatUndo).
            buttons.AddChild(SimUiKit.MakeMenuButton("Rewind",
                "⟲ Turn: undo to the start of this combat turn · ⟲ Room: restart this room (same seed) · ⟲⟲ Back: one room further back",
                new SimUiKit.MenuEntry[]
                {
                    new("⟲ Turn", "Undo to the start of the current combat turn (replays the game's combat tape)", SimActions.UndoTurn),
                    new("⟲ Room", "Rewind to the start of the current room (same seed, exact state)", SimActions.RestartRoom),
                    new("⟲⟲ Back", "Rewind one room further back", SimActions.BackOneRoom),
                }, SetHint));
            buttons.AddChild(SimUiKit.MakeMenuButton("Panels",
                "Log (F3) · HUD (F4) · Help (Home) · Settings (works mid-run)",
                new SimUiKit.MenuEntry[]
                {
                    new("Log  (F3)", "Show / hide the action log (F3)", SimActions.ToggleLog),
                    new("HUD  (F4)", "Show / hide the advice HUD (F4)", SimActions.ToggleHud),
                    new("Help  (Home)", "What everything does (Home)", SimActions.ToggleHelp),
                    new("Settings", "Mod settings -- works mid-run (the main-menu page is run-locked)", SimSettingsPanel.Instance.Toggle),
                }, SetHint));
            buttons.AddChild(SimUiKit.MakeMenuButton("Workbench",
                "Open the workbench in your browser · Copy state (Ctrl+Shift+C) to paste a position into it",
                new SimUiKit.MenuEntry[]
                {
                    new("Open workbench", "Open the workbench in your browser: logs, search tree, run tree", () => SimWorkbenchLink.Open()),
                    new("Copy state  (Ctrl+Shift+C)", "Copy this position to the clipboard -- paste it into the workbench to replay it, branch it, or sweep it in SimLab", SimActions.CopyState),
                }, SetHint));

            strip.AddChild(buttons);
            row.AddChild(strip);
            container.AddChild(row);

            // Row 2: status line (mode / advisor / last action / note).
            _statusLabel = SimUiKit.SmallLabel("", Colors.White, 15);
            _statusLabel.Visible = false;
            container.AddChild(_statusLabel);

            // Row 3: hover hint -- the owned replacement for native tooltips.
            _hintLabel = SimUiKit.SmallLabel("", new Color(0.85f, 0.88f, 0.95f));
            _hintLabel.Visible = false;
            container.AddChild(_hintLabel);

            var ticker = new Godot.Timer
            {
                Name = "SpireMentorToolbarTicker",
                WaitTime = 0.15,
                Autostart = true,
                OneShot = false,
            };
            ticker.Connect(Godot.Timer.SignalName.Timeout, Callable.From(OnTick));
            layer.AddChild(ticker);

            layer.AddChild(container);
            tree.Root.AddChildSafely(layer);

            _layer = layer;
            _container = container;
            _strip = strip;
            _buttons = buttons;
            Logger.Log("[SimToolbar] Toolbar created");
            return true;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Failed to create toolbar: {ex.Message}");
            _layer = null;
            _container = null;
            return false;
        }
    }

    private const float PanelLeft = 16f;

    // ------------------------------------------------------------------
    // Behavior (main thread only below this line)
    // ------------------------------------------------------------------

    private void TogglePill()
    {
        _expanded = !_expanded;
        SpireMentorModConfig.ToolbarExpanded = _expanded;
        SpireMentorModConfig.SaveSoon();
        Refresh();
    }

    private void SetHint(string? hint)
    {
        if (_hintLabel == null || !GodotObject.IsInstanceValid(_hintLabel))
        {
            return;
        }
        _hintLabel.Text = hint ?? "";
        _hintLabel.Visible = !string.IsNullOrEmpty(hint);
    }

    private void OnTick()
    {
        try
        {
            if (_container == null || !GodotObject.IsInstanceValid(_container))
            {
                return;
            }
            float top = SimTopBarLayout.SafeTopY(_container);
            if (Math.Abs(_container.Position.Y - top) > 0.5f)
            {
                _container.Position = new Vector2(PanelLeft, top);
            }
            // Speed and autoplay-armed raise no Changed event; poll them.
            RefreshLiveLabels();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Tick failed: {ex.Message}");
        }
    }

    private void Refresh()
    {
        if (!EnsureBuilt())
        {
            return;
        }

        SimMode mode = SimRuntime.Mode;
        if (!_modeSeen || mode != _lastMode)
        {
            // Off auto-collapses to the pill (the mod should take no space
            // while it is doing nothing); turning it on from ANY surface
            // expands the strip so the new state is visible.
            if (_modeSeen)
            {
                _expanded = mode != SimMode.Off;
            }
            _lastMode = mode;
            _modeSeen = true;
        }

        if (_strip != null && GodotObject.IsInstanceValid(_strip))
        {
            _strip.Visible = _expanded;
        }

        if (_pill != null && GodotObject.IsInstanceValid(_pill))
        {
            _pill.Text = PillText(mode);
        }

        if (_coach != null && GodotObject.IsInstanceValid(_coach))
        {
            _coach.Text = CoachStyle.PowerLabel((SimModeLite)(int)mode);
        }

        if (_style != null && GodotObject.IsInstanceValid(_style))
        {
            _style.Text = CoachStyle.StyleLabel(SimRuntime.Style);
        }

        if (_peek != null && GodotObject.IsInstanceValid(_peek))
        {
            _peek.Visible = SimRuntime.CoachStyleActive;
        }

        bool takeoverEnabled = mode == SimMode.Advisor;
        if (_aiOnce != null && GodotObject.IsInstanceValid(_aiOnce))
        {
            _aiOnce.Disabled = !takeoverEnabled;
        }
        if (_aiAuto != null && GodotObject.IsInstanceValid(_aiAuto))
        {
            _aiAuto.Disabled = !takeoverEnabled;
        }

        RefreshLiveLabels();
        RefreshStatusLine();
    }

    /// <summary>
    /// The collapsed pill's label. An engine build outranks the mode here: it
    /// is the one state where nothing works yet and the only correct action is
    /// to wait, and the pill shows collapsed — which the status line does not.
    /// </summary>
    private string PillText(SimMode mode) =>
        _advisorStatus == SimAdvisorStatus.Building
            ? $"▸ Spire Mentor — building engine… {BuildElapsed()} (leave running)"
            : _expanded
                ? "▾ Spire Mentor"
                : mode == SimMode.Off
                    ? "▸ Spire Mentor — AI off (click)"
                    : $"▸ Spire Mentor · {CoachStyle.ModeName((SimModeLite)(int)mode, SimRuntime.Style)}";

    /// <summary>Wall-clock since the engine build started, as "m:ss".</summary>
    private string BuildElapsed()
    {
        if (_buildStartedUtc is not { } started)
        {
            return "0:00";
        }
        TimeSpan d = DateTime.UtcNow - started;
        if (d < TimeSpan.Zero)
        {
            d = TimeSpan.Zero; // clock stepped back mid-build; never render "-1:59"
        }
        return $"{(int)d.TotalMinutes}:{d.Seconds:00}";
    }

    private void RefreshLiveLabels()
    {
        // The build counter has no event to fire on -- it is pure elapsed time,
        // so the 150 ms ticker is what advances it.
        if (_advisorStatus == SimAdvisorStatus.Building
            && _pill != null && GodotObject.IsInstanceValid(_pill))
        {
            string text = PillText(SimRuntime.Mode);
            if (_pill.Text != text)
            {
                _pill.Text = text;
            }
        }
        if (_speedLabel != null && GodotObject.IsInstanceValid(_speedLabel))
        {
            string speed = $"{SimSpeed.Desired:0.0}x";
            if (_speedLabel.Text != speed)
            {
                _speedLabel.Text = speed;
            }
        }
        if (_aiAuto != null && GodotObject.IsInstanceValid(_aiAuto))
        {
            string label = SimRuntime.AdvisorAutoplayArmed ? "Stop auto" : "AI auto";
            if (_aiAuto.Text != label)
            {
                _aiAuto.Text = label;
            }
        }
    }

    private void RefreshStatusLine()
    {
        if (_statusLabel == null || !GodotObject.IsInstanceValid(_statusLabel))
        {
            return;
        }
        string status = $"Spire Mentor · {SimRuntime.Mode}";
        // Autopilot armed by the config file but with nobody driving it: say
        // so, and say how to get out. Silence here is what let a human play a
        // whole run with the AI's modifiers on (2026-08-05).
        if (SimRuntime.Mode == SimMode.Autopilot && !SimRuntime.AutopilotEngaged)
        {
            status += " (idle — waiting for the AI client; Insert or 'Take over')";
        }
        if (_advisorStatus != SimAdvisorStatus.None)
        {
            status += $" · advisor: {_advisorStatus.ToString().ToLowerInvariant()}";
        }
        // The HUD toggle persists across launches; a hidden HUD with a
        // connected advisor otherwise reads as "the coach does nothing"
        // (2026-10-17, a v0.5.0 player with ShowHud saved off).
        if (SimRuntime.Mode != SimMode.Off && !SpireMentorModConfig.ShowHud)
        {
            status += " · HUD hidden (F4 shows it)";
        }
        if (_advisorStatus == SimAdvisorStatus.Building)
        {
            // Say how long and why, or the wait reads as a hang and gets killed.
            status += $"\nbuilding the Rust engine ({BuildElapsed()}) — first launch after an " +
                      "engine change, usually MINUTES. Quitting kills it and the next launch " +
                      "starts over; pre-build with `just advisor-warm` in sts2-pycli.";
        }
        if (!string.IsNullOrEmpty(_lastActionSummary))
        {
            status += $"\nlast: {_lastActionSummary}";
        }
        if (!string.IsNullOrEmpty(_note))
        {
            status += $"\n{_note}";
        }
        bool interesting = _expanded ||
            (SimRuntime.Mode == SimMode.Autopilot && !SimRuntime.AutopilotEngaged);
        _statusLabel.Text = status;
        _statusLabel.Visible = interesting && SimRuntime.Mode != SimMode.Off;
    }

    private static void RunOnMainThread(Action action)
    {
        try
        {
            if (NGame.IsMainThread())
            {
                Guarded(action);
            }
            else
            {
                Callable.From(() => Guarded(action)).CallDeferred();
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Dispatch failed: {ex.Message}");
        }
    }

    private static void Guarded(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Ignoring UI error (never breaks gameplay): {ex.Message}");
        }
    }
}
