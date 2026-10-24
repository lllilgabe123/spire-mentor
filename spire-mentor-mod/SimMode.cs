// SimMode.cs -- runtime operating mode + live settings layer.
//
// SimConfig (SpireMentor.conf) stays the authoritative, immutable, file-based
// configuration: the parity/win-rate sweep lane drives the mod purely through
// that file and must keep behaving identically. SimRuntime sits on top of it
// as the LIVE state -- initialised from the file at startup, mutable
// afterwards (BaseLib settings page, hotkeys, console commands), with a
// Changed event for the HUD/advisor to react to.
//
// Mode semantics:
//   Off       -- the mod is inert beyond the TCP server: no Harmony behavior
//                changes, no HUD, the game plays 100% vanilla.
//   Advisor   -- the HUMAN plays; the mod observes decision points, asks the
//                Python advisor for a ranking, and renders it -- before the
//                player acts in the Hints style, only on F5 / afterwards (the
//                coach chip) in the Coach style (SimRuntime.Style). Must never
//                enable NonInteractiveMode/AutoSlay paths: with
//                AutoSlayerCheck active, Cmd.ScaledWait skips all waits and
//                CombatManager.Pause() no-ops -- unplayable for a human.
//   Autopilot -- the AI plays the whole run via SimAutoSlayer (the original
//                bridge behavior; what every sweep/eval harness uses).
//
// Autopilot is ARMED by the config file but only ENGAGES once an agent is
// actually connected (see AutopilotEngaged) -- until then the game stays
// completely vanilla, so booting with auto_start=true and playing by hand is
// safe. Leaving Autopilot at runtime (F8 / console / settings) stops the
// SimAutoSlayer rather than being refused.
//
// Startup mapping preserves every existing invocation: auto_start=true (the
// sweep lane) => Autopilot, auto_start=false => Off. A human launch then
// flips Off -> Advisor once the game is up if the persisted CoachOnAtLaunch
// setting allows it (MainFile.CoachOnWhenReadyAsync) -- never from
// Autopilot, never under advisor_autoplay.

using System;
using System.Threading;

namespace SpireMentor;

public enum SimMode
{
    Off,
    Advisor,
    Autopilot,
}

public static class SimRuntime
{
    /// <summary>
    /// Raised after any SimRuntime property changes. May fire on ANY thread
    /// (settings UI = main thread, bridge/console = worker threads);
    /// subscribers that touch the scene tree must marshal via CallDeferred.
    /// </summary>
    public static event Action? Changed;

    private static int _mode; // int-backed for Interlocked; values are SimMode

    /// <summary>
    /// The live operating mode. The gated Harmony patches (IsReleaseGamePatch,
    /// WaitSpeedPatch, AnimationSpeedPatch) consult this on every call, so
    /// flipping it takes effect immediately without re/un-patching. Note the
    /// one-way ratchets that a flip does NOT undo: an already-started
    /// SimAutoSlayer keeps running until it stops itself, and anything the
    /// game latched while IsReleaseGame() returned false stays latched.
    /// </summary>
    public static SimMode Mode
    {
        get => (SimMode)Volatile.Read(ref _mode);
        set
        {
            if (Interlocked.Exchange(ref _mode, (int)value) == (int)value)
            {
                return;
            }
            RaiseChanged();
        }
    }

    private static int _style = (int)AdviceStyle.Coach; // values are AdviceStyle

    /// <summary>
    /// The advice style inside Advisor mode (CoachStyle.cs): Hints renders
    /// the AI's opinion before the player acts, Coach renders nothing until
    /// they have (F5 peeks). The advisor computes advice either way, so F6/F7
    /// and the unattended drivers -- which read SimAdvisor's standing
    /// recommendation, never the HUD -- work the same in both.
    /// </summary>
    public static AdviceStyle Style
    {
        get => (AdviceStyle)Volatile.Read(ref _style);
        set
        {
            if (Interlocked.Exchange(ref _style, (int)value) == (int)value)
            {
                return;
            }
            RaiseChanged();
        }
    }

    /// <summary>Advisor mode in the Coach style: every pre-action advice
    /// surface (panel, eval bar, line, card scores, map markers, toolbar
    /// note) stays empty.</summary>
    public static bool CoachStyleActive => Mode == SimMode.Advisor && Style == AdviceStyle.Coach;

    /// <summary>Set mode and style together, raising Changed at most once
    /// per actual change.</summary>
    public static void SetModeAndStyle(SimMode mode, AdviceStyle style)
    {
        // Style first: the mode flip is what subscribers react to, and it
        // must already see the style it lands in.
        bool styleChanged = Interlocked.Exchange(ref _style, (int)style) != (int)style;
        bool modeChanged = Interlocked.Exchange(ref _mode, (int)mode) != (int)mode;
        if (styleChanged || modeChanged)
        {
            RaiseChanged();
        }
    }

    /// <summary>
    /// True only while autopilot is ACTUALLY driving: the mode is Autopilot,
    /// a SimAutoSlayer is running, and an agent is connected to answer it.
    ///
    /// Everything that changes how the game feels -- IsReleaseGamePatch,
    /// WaitSpeedPatch, AnimationSpeedPatch and NonInteractiveMode.
    /// AutoSlayerCheck -- gates on THIS, not on Mode alone. Mode alone was a
    /// bug: `auto_start=true` is a persistent setting in SpireMentor.conf, so
    /// every human session launched after a sweep booted straight into
    /// Autopilot, and SimAutoSlayer.Start() flipped the whole game
    /// non-interactive while it sat blocked in SimAgentGate waiting for a
    /// client that a human player never starts. With waits skipped
    /// (Cmd.ScaledWait) the human's own run desynchronised -- observed
    /// 2026-08-05: the Neow "transform a card" blessing resolved against a
    /// half-dealt deck, auto-picking a Strike with no selection screen.
    ///
    /// A sweep is unaffected: its client is connected for the whole session.
    /// </summary>
    public static bool AutopilotEngaged =>
        Mode == SimMode.Autopilot &&
        !SimMultiplayer.InRun &&
        SimAutoSlayer.IsActive &&
        BridgeServer.Instance.IsClientConnected;

    private static int _advisorAutoplayArmed; // int-backed bool for Volatile

    /// <summary>
    /// True while advisor-mode autoplay (F7) is armed. Written by SimAdvisor
    /// on every arm/disarm so the speed patches below can consult it without
    /// reaching into SimAdvisor's private state.
    /// </summary>
    public static bool AdvisorAutoplayArmed
    {
        get => Volatile.Read(ref _advisorAutoplayArmed) != 0;
        set => Volatile.Write(ref _advisorAutoplayArmed, value ? 1 : 0);
    }

    /// <summary>
    /// The advisor lane's UNATTENDED shape: autoplay armed in Advisor mode
    /// under an `advisor_autoplay` (harness) config. Nobody is watching, so
    /// the human-pacing rules do not apply. The `advisor_autoplay` gate is
    /// what keeps a human's own F7 session at human speed: that flag is
    /// default-false, never persisted by the settings page, and only ever
    /// written by the unattended drivers (ascension climb, F7 harness).
    /// </summary>
    public static bool UnattendedAdvisorActive =>
        Mode == SimMode.Advisor &&
        AdvisorAutoplayArmed &&
        SimConfig.Instance.AdvisorAutoplay;

    /// <summary>
    /// Gate for WaitSpeedPatch/AnimationSpeedPatch: every context where the
    /// game's pacing exists only to please an audience that is not there.
    /// Historically AutopilotEngaged alone — which left the unattended
    /// advisor-autoplay lane (climb / F7 harness) animating at HUMAN speed:
    /// measured 2026-08-10 over climb runs 029-034, ~58% of run wall-clock
    /// was game pacing, ~1.1 s per combat decision.
    /// </summary>
    public static bool SpeedupsEngaged => AutopilotEngaged || UnattendedAdvisorActive;

    private static float _waitMultiplier = 0.1f;
    private static float _animMultiplier = 5.0f;

    /// <summary>
    /// Live value behind WaitSpeedPatch (Cmd.CustomScaledWait scaling).
    /// Default 0.1f -- the value every sweep has always run with. Only
    /// consulted while Mode == Autopilot. Mutable via console
    /// (`spirementor waitmult`); deliberately NOT on the settings page so a
    /// persisted UI value can never silently retune the sweep lane.
    /// </summary>
    public static float WaitMultiplier
    {
        get => Volatile.Read(ref _waitMultiplier);
        set { Volatile.Write(ref _waitMultiplier, value); RaiseChanged(); }
    }

    /// <summary>Live value behind AnimationSpeedPatch. Default 5.0f. Same
    /// rules as <see cref="WaitMultiplier"/>.</summary>
    public static float AnimMultiplier
    {
        get => Volatile.Read(ref _animMultiplier);
        set { Volatile.Write(ref _animMultiplier, value); RaiseChanged(); }
    }

    /// <summary>
    /// One-time startup initialisation from the config file. Sets fields
    /// directly (no Changed event -- nothing is subscribed yet, and startup
    /// must not look like a user action).
    /// </summary>
    public static void InitializeFrom(SimConfig config)
    {
        _mode = (int)(config.AutoStart ? SimMode.Autopilot : SimMode.Off);
        // The persisted default style (settings: "start in Coach style"). The
        // launch path re-reads it once BaseLib has loaded the saved value
        // (MainFile.CoachOnWhenReadyAsync).
        // Guarded: the settings class derives from a BaseLib type, and a
        // missing BaseLib must not take the boot down with it.
        try
        {
            _style = (int)ReadDefaultStyle();
        }
        catch (Exception)
        {
            _style = (int)CoachStyle.FromDefaultFlag(CoachStyle.CoachByDefault);
        }
    }

    /// <summary>Separate and never inlined, so a BaseLib type-load failure
    /// surfaces inside the caller's try instead of at its JIT.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static AdviceStyle ReadDefaultStyle() => SpireMentorModConfig.DefaultAdviceStyle;

    /// <summary>
    /// Raise Changed for live state that lives outside SimRuntime (e.g. a
    /// hotkey flipped a HUD toggle on SpireMentorModConfig) so the HUD and an
    /// open settings page refresh.
    /// </summary>
    public static void NotifyChanged() => RaiseChanged();

    private static void RaiseChanged()
    {
        try
        {
            Changed?.Invoke();
        }
        catch (Exception ex)
        {
            Godot.GD.PrintErr($"[SpireMentor] SimRuntime.Changed handler threw: {ex}");
        }
    }
}
