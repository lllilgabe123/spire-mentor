// SimCoachPanel.cs -- the fight-recap and death-autopsy panel: a collapsible,
// non-modal block on the HUD layer (PROTOCOL.md section 5, v5.1 `recap` /
// `autopsy`; the words come from CoachPanelText).
//
// Shown in both advice styles: Hints also gets after-the-fact feedback.
//
// Behaviour:
//   * one panel at a time; a new recap / autopsy replaces the shown one;
//   * right-aligned under the toolbar, max 460 px wide, scrolls when taller
//     than 55 % of the viewport, so it stays clear of the reward list, the
//     Proceed button and the game-over screen's buttons;
//   * the header's ▾/▸ collapses it to title + summary, × closes it; every
//     section with a run-log step gets an "open in workbench" button;
//   * a recap hides when the next room starts (the player's next combat turn,
//     or a screen decision other than the rewards / card reward / map); an
//     autopsy hides once the game-over screen is gone; a `withdraw` naming
//     the panel's id hides it.
//
// Input-safety contract (SimUiKit.cs): the layer's containers ignore the
// mouse, the panel body stops it (and the scroll area scrolls), buttons stop
// their own clicks, nothing takes focus. Main thread only; SimAdvisor
// marshals.

using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace SpireMentor;

internal sealed class SimCoachPanel
{
    public static readonly SimCoachPanel Instance = new();

    private const float PanelWidth = 460f;
    private const float RightMargin = 16f;
    private const float MaxHeightShare = 0.55f;
    private const float FallbackTop = 80f;

    private CanvasLayer? _layer;
    private PanelContainer? _panel;
    private ScrollContainer? _scroll;
    private VBoxContainer? _body;
    private Button? _collapse;
    private PanelText? _shown;
    private bool _collapsed;

    private SimCoachPanel()
    {
        SimRuntime.Changed += () => Callable.From(RefreshVisibility).CallDeferred();
    }

    // ------------------------------------------------------------------
    // Entry points (main thread)
    // ------------------------------------------------------------------

    public void Show(PanelText text)
    {
        Guard(() =>
        {
            if (!EnsureLayer())
            {
                return;
            }
            Clear();
            _shown = text;
            _collapsed = false;
            Build(text);
            RefreshVisibility();
        });
    }

    /// <summary>Hide the panel if it shows one of <paramref name="ids"/>.</summary>
    public void Withdraw(IReadOnlyCollection<string> ids) =>
        Guard(() =>
        {
            if (_shown != null && ids.Contains(_shown.Id))
            {
                Clear();
            }
        });

    /// <summary>The player's next combat turn started: a recap is over.</summary>
    public void OnCombatTurn() =>
        Guard(() =>
        {
            if (_shown?.Kind == "recap")
            {
                Clear();
            }
        });

    /// <summary>Every screen poll: <paramref name="adviceFor"/> is the decision
    /// on screen (null = none). A recap survives the rewards, the card reward
    /// and the map; an autopsy only the game-over screen.</summary>
    public void OnScreen(string? adviceFor) =>
        Guard(() =>
        {
            if (_shown == null)
            {
                return;
            }
            bool keep = _shown.Kind switch
            {
                "recap" => adviceFor is null or "reward_screen" or "card_reward" or "map_select",
                "autopsy" => adviceFor == "game_over",
                _ => false,
            };
            if (!keep)
            {
                Clear();
            }
        });

    /// <summary>The run rewound or a new run started: nothing to recap.</summary>
    public void Hide() => Guard(Clear);

    // ------------------------------------------------------------------

    private void Clear()
    {
        _shown = null;
        if (_panel != null && GodotObject.IsInstanceValid(_panel))
        {
            _panel.QueueFree();
        }
        _panel = null;
        _scroll = null;
        _body = null;
        _collapse = null;
    }

    private void Build(PanelText text)
    {
        StyleBoxFlat style = SimUiKit.PanelStyle();
        var panel = new PanelContainer
        {
            Name = "SpireMentorCoachPanel",
            CustomMinimumSize = new Vector2(PanelWidth, 0f),
            MouseFilter = Control.MouseFilterEnum.Stop,
            FocusMode = Control.FocusModeEnum.None,
        };
        panel.AddThemeStyleboxOverride("panel", style);

        var rows = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        rows.AddThemeConstantOverride("separation", 4);

        var header = new HBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        Label title = SimUiKit.SmallLabel(text.Title, new Color(1f, 0.85f, 0.4f), 16);
        title.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        title.ClipText = true;
        header.AddChild(title);
        _collapse = SimUiKit.MakeButton("▾", "", () => Guard(ToggleCollapsed));
        _collapse.AddThemeFontSizeOverride("font_size", 12);
        header.AddChild(_collapse);
        Button close = SimUiKit.MakeButton("×", "", () => Guard(Clear));
        close.AddThemeFontSizeOverride("font_size", 12);
        header.AddChild(close);
        rows.AddChild(header);

        if (text.Summary.Length > 0)
        {
            rows.AddChild(Wrapped(text.Summary, new Color(0.92f, 0.94f, 0.98f), 14));
        }

        var body = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        body.AddThemeConstantOverride("separation", 6);
        foreach (PanelSection section in text.Sections)
        {
            body.AddChild(BuildSection(section, text.LogName));
        }
        if (text.Footer.Length > 0)
        {
            body.AddChild(Wrapped(text.Footer, new Color(0.7f, 0.72f, 0.78f), 12));
        }

        // Scrolls only when the content is taller than the cap (Tick sizes it).
        var scroll = new ScrollContainer
        {
            MouseFilter = Control.MouseFilterEnum.Pass,
            FocusMode = Control.FocusModeEnum.None,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        scroll.AddChild(body);
        rows.AddChild(scroll);
        panel.AddChild(rows);

        _layer!.AddChild(panel);
        _panel = panel;
        _scroll = scroll;
        _body = body;
        Place();
    }

    private Control BuildSection(PanelSection section, string? logName)
    {
        var box = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        box.AddThemeConstantOverride("separation", 1);
        box.AddChild(Wrapped(section.Heading, new Color(1f, 0.78f, 0.45f), 14));
        foreach (string line in section.Lines)
        {
            box.AddChild(Wrapped(line, new Color(0.88f, 0.9f, 0.95f), 13));
        }
        // Its own row under the section: next to a full-width heading it
        // pushed the panel past PanelWidth and off the right screen edge.
        if (section.LogIdx is int step)
        {
            var row = new HBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
            Button open = SimUiKit.MakeButton("open in workbench", "",
                () => Guard(() => SimWorkbenchLink.Open(logName, step)));
            open.AddThemeFontSizeOverride("font_size", 11);
            open.SizeFlagsHorizontal = Control.SizeFlags.ShrinkBegin;
            row.AddChild(open);
            box.AddChild(row);
        }
        return box;
    }

    private static Label Wrapped(string text, Color color, int size)
    {
        Label l = SimUiKit.SmallLabel(text, color, size);
        l.CustomMinimumSize = new Vector2(PanelWidth - 40f, 0f);
        l.AutowrapMode = TextServer.AutowrapMode.WordSmart;
        return l;
    }

    private void ToggleCollapsed()
    {
        _collapsed = !_collapsed;
        if (_scroll != null && GodotObject.IsInstanceValid(_scroll))
        {
            _scroll.Visible = !_collapsed;
        }
        if (_collapse != null && GodotObject.IsInstanceValid(_collapse))
        {
            _collapse.Text = _collapsed ? "▸" : "▾";
        }
    }

    /// <summary>Right-align under the toolbar and cap the scroll height
    /// (every tick: the viewport and the toolbar can move).</summary>
    private void Place()
    {
        // Not yet in the tree on the first Show: the layer is added deferred
        // (EnsureLayer), and the ticker places the panel once it is.
        if (_panel == null || !GodotObject.IsInstanceValid(_panel) || !_panel.IsInsideTree())
        {
            return;
        }
        Vector2 view = _panel.GetViewportRect().Size;
        float top = (SimToolbar.Instance.BottomY() ?? FallbackTop) + 8f;
        if (_scroll != null && _body != null && GodotObject.IsInstanceValid(_body))
        {
            float cap = Math.Max(120f, view.Y * MaxHeightShare);
            float want = Math.Min(_body.GetCombinedMinimumSize().Y, cap);
            if (Math.Abs(_scroll.CustomMinimumSize.Y - want) > 0.5f)
            {
                _scroll.CustomMinimumSize = new Vector2(PanelWidth - 20f, want);
            }
        }
        // The real width, not PanelWidth: a long unbreakable word can still
        // widen the panel, and it must grow leftwards, never off-screen.
        float width = Math.Max(PanelWidth, _panel.Size.X);
        var pos = new Vector2(Math.Max(0f, view.X - width - RightMargin), top);
        if (_panel.Position.DistanceTo(pos) > 0.5f)
        {
            _panel.Position = pos;
        }
    }

    private void RefreshVisibility()
    {
        Guard(() =>
        {
            if (_layer != null && GodotObject.IsInstanceValid(_layer))
            {
                _layer.Visible = SpireMentorModConfig.ShowHud && SimRuntime.Mode == SimMode.Advisor;
            }
        });
    }

    private bool EnsureLayer()
    {
        if (_layer != null && GodotObject.IsInstanceValid(_layer))
        {
            return true;
        }
        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return false;
        }
        // The HUD's layer number (99), as its own CanvasLayer like the chip.
        var layer = new CanvasLayer { Name = "SpireMentorCoachPanelLayer", Layer = 99 };
        var ticker = new Godot.Timer { Name = "CoachPanelTicker", WaitTime = 0.25, Autostart = true, OneShot = false };
        ticker.Connect(Godot.Timer.SignalName.Timeout, Callable.From(() => Guard(Place)));
        layer.AddChild(ticker);
        // Deferred: Show can run while the tree is busy (a coach record lands
        // between frames).
        Callable.From(() => tree.Root.AddChild(layer)).CallDeferred();
        _layer = layer;
        SimCoachChip.Log("[SimCoachPanel] panel layer created");
        return true;
    }

    private static void Guard(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            SimCoachChip.Log($"[SimCoachPanel] Ignoring error (never breaks gameplay): {ex.Message}");
        }
    }
}
