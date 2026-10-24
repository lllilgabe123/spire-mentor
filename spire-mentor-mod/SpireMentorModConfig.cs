// SpireMentorModConfig.cs -- BaseLib settings page for SpireMentor.
//
// Reachable from the main menu via BaseLib's "Mod Configuration" entry
// (BaseLib deliberately disables that entry while a run is in progress --
// ModConfigPatch.NSettingsScreen_OnSubmenuShown_Patch -- so mid-run control
// goes through SimHotkeys / the `spirementor` console command instead).
//
// Two kinds of options live here:
//
//   * [ConfigIgnore] LIVE PROXIES (AdvisorMode, GameSpeed) -- read/write the
//     runtime state directly and are never written to the BaseLib config
//     file. This is deliberate: SpireMentor.conf is the boot authority for the
//     sweep lane, and a persisted "advisor" or "0.5x speed" from a human
//     session must never leak into the next headless sweep.
//   * Persisted preferences (HUD toggles, advisor-launch settings) -- normal
//     BaseLib persistence to user://mod_configs/SpireMentor.cfg; they have no
//     effect on autopilot behavior.
//
// UI rows are hand-built in SetupConfigUI (instead of SimpleModConfig's
// auto-generation) because label/section text otherwise falls back to raw
// property names: BaseLib's loc lookup needs a settings_ui table shipped in
// a .pck, and SpireMentor deliberately has no .pck (see SpireMentor.csproj).

using System;
using System.Reflection;
using BaseLib.Config;
using BaseLib.Config.UI;
using BaseLib.Extensions;
using Godot;
using MegaCrit.Sts2.addons.mega_text;

namespace SpireMentor;

public sealed class SpireMentorModConfig : SimpleModConfig
{
    private static SpireMentorModConfig? _instance;

    // ------------------------------------------------------------------
    // Live proxies (never persisted).
    // ------------------------------------------------------------------

    /// <summary>
    /// Off &lt;-&gt; Advisor toggle. Turning it on while the mod booted in
    /// Autopilot (auto_start=true) STOPS the autopilot session and hands the
    /// game back -- see MainFile.StopAutopilot. Autopilot itself can only be
    /// entered at launch, from the config file.
    /// </summary>
    [ConfigIgnore]
    public static bool AdvisorMode
    {
        get => SimRuntime.Mode == SimMode.Advisor;
        set
        {
            if (SimRuntime.Mode == SimMode.Autopilot)
            {
                MainFile.StopAutopilot();
            }
            if (value && SimRuntime.Mode != SimMode.Advisor)
            {
                SimRuntime.SetModeAndStyle(SimMode.Advisor, DefaultAdviceStyle);
                return;
            }
            SimRuntime.Mode = value ? SimMode.Advisor : SimMode.Off;
        }
    }

    /// <summary>
    /// Whole-game speed (Engine.TimeScale), live. Range copies the game's
    /// own debug clamp. Reading shows the actual current speed; writing
    /// applies immediately. Never persisted, never auto-applied at boot.
    /// </summary>
    [ConfigIgnore]
    [ConfigSlider(0.1, 4.0, 0.1, Format = "{0:0.0}x")]
    public static double GameSpeed
    {
        get => SimSpeed.Current;
        set => SimSpeed.Apply(value);
    }

    // ------------------------------------------------------------------
    // Persisted preferences.
    // ------------------------------------------------------------------

    public static bool ShowHud { get; set; } = true;
    public static bool ShowRankedList { get; set; } = true;
    public static bool ShowCardLabels { get; set; } = true;

    /// <summary>Map-node visit markers + projected route. Used to ride on
    /// ShowCardLabels (a naming mismatch users kept tripping over).</summary>
    public static bool ShowMapMarkers { get; set; } = true;

    public static bool ShowWinBar { get; set; } = true;

    /// <summary>Skip purely decorative combat effects (card trails, buff and
    /// debuff flashes) -- see SimReduceEffects. Visual only; takes effect for
    /// the next effect spawned.</summary>
    public static bool ReduceEffects { get; set; } = false;

    public static bool AdvisorAutoLaunch { get; set; } = true;

    /// <summary>Plan fights with the true draw order and RNG cursors the mod
    /// already reads, instead of the deployed blindfolded search (the
    /// advisor's --fight-sight). Read when the advisor starts.</summary>
    public static bool ClairvoyantFights { get; set; } = true;

    /// <summary>Switch the coach on by itself at launch (see
    /// MainFile.CoachOnWhenReadyAsync), so enabling the mod in the game's mod
    /// list is the only step. Only from Off and only on Windows: Autopilot
    /// (auto_start, the sweep lane) and the advisor_autoplay harness keep
    /// their own startup paths, and the bundled advisor cannot start
    /// elsewhere.</summary>
    public static bool CoachOnAtLaunch { get; set; } = true;

    /// <summary>
    /// Persisted backing for <see cref="DefaultAdviceStyle"/>: true = Coach
    /// (nothing before you act), false = Hints. A tickbox because that is the
    /// control kind this page builds; the style is what the code reads.
    /// </summary>
    public static bool StartInCoachStyle { get; set; } = CoachStyle.CoachByDefault;

    /// <summary>The style the coach turns on in at launch (and from the
    /// settings tickbox). Default for a fresh config: Coach.</summary>
    [ConfigIgnore]
    public static AdviceStyle DefaultAdviceStyle => CoachStyle.FromDefaultFlag(StartInCoachStyle);

    /// <summary>The coach chip also confirms good turns and agreed choices
    /// with a small ✓ (default off: silence on good play).</summary>
    public static bool CoachShowGoodTurns { get; set; } = false;

    /// <summary>Toolbar strip expanded/collapsed (SimToolbar pill).</summary>
    public static bool ToolbarExpanded { get; set; } = true;

    /// <summary>Workbench page the toolbar's button opens. Blank = the
    /// bundled workbench the advisor hosts, else the dev one on :8077 (see
    /// SimWorkbenchLink).</summary>
    public static string WorkbenchUrl { get; set; } = "";

    /// <summary>Command line used to launch the Python advisor process.
    /// Empty = the built-in default (see SimAdvisorProcess, phase 8).</summary>
    public static string AdvisorCommand { get; set; } = "";

    /// <summary>Web origin of a BROWSER workbench allowed to reach the
    /// advisor's control plane (the live-game tree, live search and rewind
    /// of the browser-only build, GH #709) -- e.g.
    /// <c>https://your-workbench.example.com</c>. Blank = off: no web page can
    /// read the control plane, only the same-machine workbench proxy. Passed
    /// to the advisor as <c>--control-allow-origin</c>.</summary>
    public static string BrowserWorkbenchOrigin { get; set; } = "";

    // ------------------------------------------------------------------

    /// <summary>Debounced persist for edits made outside the BaseLib page
    /// (toolbar pill, mid-run settings panel) -- BaseLib itself only saves on
    /// submenu close and on quit.</summary>
    public static void SaveSoon() => _instance?.SaveDebounced();

    /// <summary>
    /// Construct + register with BaseLib, and keep an open settings page in
    /// sync when the mode changes from a hotkey or console command.
    /// Called once from MainFile.Initialize; failures there are non-fatal.
    /// </summary>
    public static void Register()
    {
        _instance = new SpireMentorModConfig();
        ModConfigRegistry.Register(MainFile.ModId, _instance);

        SimRuntime.Changed += () =>
        {
            SpireMentorModConfig? cfg = _instance;
            if (cfg == null)
            {
                return;
            }
            // May fire on any thread; control refreshes touch the scene tree.
            Callable.From(cfg.ConfigReloaded).CallDeferred();
        };
    }

    public override void SetupConfigUI(Control optionContainer)
    {
        optionContainer.AddChild(CreateSectionHeader("Spire Mentor AI", alignToTop: true));
        // The page opens on a mod that, by design, is doing nothing yet. Say
        // what turning it on gets you before offering the switch.
        AddNote(optionContainer,
            "The AI coach has two styles. Hints shows its opinion while you decide: a score on\n" +
            "each card, a ranked list of moves, a win-chance bar and the route it likes on the\n" +
            "map. Coach shows nothing before you act -- press F5 to peek at its next pick -- and\n" +
            "afterwards a small note says where your turn or choice differed from its own.\n" +
            "Insert turns it on / off; the toolbar's Hints / Coach button switches style.\n" +
            "It switches on when the game starts (untick\n" +
            "that below if you would rather press Insert), and it never plays for you unless you\n" +
            "ask it to (Ctrl+Shift+P).");
        AddRow(optionContainer, "Advisor mode (AI coach)", nameof(AdvisorMode), CreateRawTickboxControl);
        AddRow(optionContainer, "Turn the coach on when the game starts", nameof(CoachOnAtLaunch), CreateRawTickboxControl);
        AddRow(optionContainer, "Start in Coach style (untick: Hints)", nameof(StartInCoachStyle), CreateRawTickboxControl);
        AddRow(optionContainer, "Coach: also mark good turns with ✓", nameof(CoachShowGoodTurns), CreateRawTickboxControl);
        AddRow(optionContainer, "Game speed", nameof(GameSpeed), CreateRawSliderControl);

        optionContainer.AddChild(CreateSectionHeader("HUD"));
        AddRow(optionContainer, "Show HUD", nameof(ShowHud), CreateRawTickboxControl);
        AddRow(optionContainer, "Show ranked moves", nameof(ShowRankedList), CreateRawTickboxControl);
        AddRow(optionContainer, "Show card scores", nameof(ShowCardLabels), CreateRawTickboxControl);
        AddRow(optionContainer, "Show map markers + route", nameof(ShowMapMarkers), CreateRawTickboxControl);
        AddRow(optionContainer, "Show win chance bar", nameof(ShowWinBar), CreateRawTickboxControl);
        AddRow(optionContainer, "Reduce combat effects", nameof(ReduceEffects), CreateRawTickboxControl);

        optionContainer.AddChild(CreateSectionHeader("Advisor backend"));
        AddRow(optionContainer, "Auto-launch Python advisor", nameof(AdvisorAutoLaunch), CreateRawTickboxControl);
        AddRow(optionContainer, "See the future in fights (draw order)", nameof(ClairvoyantFights), CreateRawTickboxControl);
        AddRow(optionContainer, "Advisor command (blank = default)", nameof(AdvisorCommand), CreateRawLineEditControl);

        // BaseLib disables this whole page during a run, so the keys are the
        // only mid-run control there is -- and they were documented nowhere a
        // player would look. F1 shows the same list in-game.
        optionContainer.AddChild(CreateSectionHeader("Keys (press Home in game for this list)"));
        AddNote(optionContainer,
            "Insert   AI on / off (from autopilot: take the controls back)\n" +
            "F5   Coach style: peek at the AI's next pick\n" +
            "Ctrl+Shift+N   let the AI take just this one decision\n" +
            "Ctrl+Shift+P   let the AI keep playing until you stop it\n" +
            "F4   show / hide the HUD          F3   action log\n" +
            "F9 / F10   game speed down / up   Home   help");

        AddRestoreDefaultsButton(optionContainer);
        SetupFocusNeighbors(optionContainer);
    }

    // ------------------------------------------------------------------
    // Mid-run settings panel (SimSettingsPanel)
    // ------------------------------------------------------------------

    /// <summary>
    /// Build the mid-run panel's rows into <paramref name="container"/>
    /// (everyday settings) and <paramref name="advanced"/> (advisor and
    /// workbench plumbing most players never touch), using the same NConfig*
    /// controls as the main-menu page (they bind the static properties and
    /// self-refresh on ConfigReloaded, so every surface stays in sync).
    /// Returns false if the control factory failed -- the caller renders a
    /// pointer to the main-menu page instead.
    /// </summary>
    internal static bool BuildLiveRowsInto(Control container, Control advanced)
    {
        SpireMentorModConfig? cfg = _instance;
        if (cfg == null)
        {
            return false;
        }
        try
        {
            cfg.AddLiveRow(container, "AI coach (advisor mode)", nameof(AdvisorMode), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(container, "Coach on when the game starts", nameof(CoachOnAtLaunch), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(container, "Start in Coach style (untick: Hints)", nameof(StartInCoachStyle), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(container, "Coach: also mark good turns with ✓", nameof(CoachShowGoodTurns), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(container, "Game speed", nameof(GameSpeed), cfg.CreateRawSliderControl);
            cfg.AddLiveRow(container, "Show HUD", nameof(ShowHud), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(container, "Show ranked moves", nameof(ShowRankedList), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(container, "Show card scores", nameof(ShowCardLabels), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(container, "Show map markers + route", nameof(ShowMapMarkers), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(container, "Show win chance bar", nameof(ShowWinBar), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(container, "Reduce combat effects", nameof(ReduceEffects), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(advanced, "Auto-launch Python advisor", nameof(AdvisorAutoLaunch), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(advanced, "See the future in fights (draw order)", nameof(ClairvoyantFights), cfg.CreateRawTickboxControl);
            cfg.AddLiveRow(advanced, "Advisor command (blank = default)", nameof(AdvisorCommand), cfg.CreateRawLineEditControl);
            cfg.AddLiveRow(advanced, "Workbench URL (blank = default)", nameof(WorkbenchUrl), cfg.CreateRawLineEditControl);
            cfg.AddLiveRow(advanced, "Browser workbench origin (blank = off)", nameof(BrowserWorkbenchOrigin), cfg.CreateRawLineEditControl);
            return true;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SpireMentorModConfig] BuildLiveRowsInto failed: {ex.Message}");
            return false;
        }
    }

    private void AddLiveRow(Control container, string labelText, string propertyName,
        Func<PropertyInfo, Control> controlCreator)
    {
        PropertyInfo property = typeof(SpireMentorModConfig).GetProperty(propertyName)
            ?? throw new InvalidOperationException($"No such config property: {propertyName}");
        Control control = controlCreator(property);
        var row = new HBoxContainer();
        row.AddThemeConstantOverride("separation", 8);
        MegaRichTextLabel label = CreateRawLabelControl(labelText, 18);
        label.FitContent = true;
        label.CustomMinimumSize = new Godot.Vector2(280f, 0f);
        label.SizeFlagsVertical = Control.SizeFlags.ShrinkCenter;
        row.AddChild(label);
        row.AddChild(control);
        container.AddChild(row);
    }

    private static bool _livePersistHooked;

    /// <summary>
    /// BaseLib only saves configs on settings-submenu close and on quit; the
    /// toolbar and the mid-run panel edit outside that flow, so hook every
    /// change into a debounced save. Idempotent.
    /// </summary>
    internal static void EnsureLivePersistHook()
    {
        SpireMentorModConfig? cfg = _instance;
        if (cfg == null || _livePersistHooked)
        {
            return;
        }
        _livePersistHooked = true;
        cfg.ConfigChanged += (_, _) => cfg.SaveDebounced();
    }

    /// <summary>
    /// A row of explanatory text with no control attached -- the page has to
    /// carry its own documentation, since SpireMentor ships no .pck and so has
    /// no loc table for tooltips.
    /// </summary>
    private void AddNote(Control container, string text)
    {
        MegaRichTextLabel label = CreateRawLabelControl(text, 20);
        // The helper builds rows for NConfigOptionRow, which sizes its label
        // for it; standing on its own in the VBox a multi-line label would
        // collapse to zero height, so it has to size to its own content.
        label.FitContent = true;
        label.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        container.AddChild(label);
        container.AddChild(CreateDividerControl());
    }

    /// <summary>
    /// CreateStandardOption with explicit label text instead of the
    /// loc-or-property-name fallback.
    /// </summary>
    private void AddRow(Control container, string labelText, string propertyName,
        Func<PropertyInfo, Control> controlCreator)
    {
        PropertyInfo property = typeof(SpireMentorModConfig).GetProperty(propertyName)
            ?? throw new InvalidOperationException($"No such config property: {propertyName}");
        Control control = controlCreator(property);
        Control label = CreateRawLabelControl(labelText, 28);
        var row = new NConfigOptionRow(ModPrefix, property.Name, label, control);
        control.ClearFocusNeighbors();
        container.AddChild(row);
        container.AddChild(CreateDividerControl());
    }
}
