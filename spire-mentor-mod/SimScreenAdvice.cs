// SimScreenAdvice.cs -- describe (and drive) the non-combat decision screen
// a HUMAN is looking at, for advisor mode.
// Originally ported from zhiyue/sts2-rl-agent (STS2BridgeMod); rewritten clean-room.
//
// Autopilot never needs this: SimCardSelector/CardSelectCmd.Selector bypasses
// these screens entirely, so the AI answers an ICardSelector callback and no
// UI exists. In advisor mode the opposite is true -- the real screen is up and
// the player is staring at it -- so advice has to be built FROM the live
// screen nodes, and "AI takes this decision" has to click them.
//
// Screen -> click recipes are the ones the game's own AutoSlay handlers use
// (v0.107.1 decomp):
//   NCardRewardSelectionScreen: UiHelper.FindAll<NCardHolder> +
//     holder.EmitSignal(NCardHolder.SignalName.Pressed, holder)
//     (SimCardRewardScreenHandler.cs, verified in live sweeps)
//   NCardGridSelectionScreen and subclasses (simple select, deck upgrade /
//     transform / enchant, choose-a-card): UiHelper.FindFirst<NCardGrid> +
//     grid.EmitSignal(NCardGrid.SignalName.HolderPressed, holder), then
//     %Confirm if the screen requires manual confirmation
//     (SimpleCardSelectScreenHandler.cs:36-70).

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace SpireMentor;

/// <summary>
/// A decision the player is being asked to make on a screen: what to send the
/// advisor, and how to execute the answer.
/// </summary>
internal sealed class SimScreenDecision
{
    /// <summary>Wire `advice_for` value ("card_reward" / "card_select").</summary>
    public string AdviceFor = "";

    /// <summary>The advice_request payload (minus type/advice_for).</summary>
    public Dictionary<string, object> State = new();

    /// <summary>Number of offered options, for validating a recommendation.</summary>
    public int OptionCount;

    /// <summary>
    /// The advisor-visible label of each option, in the SAME order the
    /// options were sent, spelled exactly as the Python builder spells it
    /// (card `Title` for the card screens, `type@(col,row)` for the map).
    ///
    /// This is the takeover's cross-check: an index alone is only as good as
    /// the assumption that the option list has not been re-ordered between
    /// describe and click, and on the reward screen that assumption is
    /// genuinely unsafe -- `SetOptions` QueueFrees the previous holders and
    /// adds the new ones in the same call (NCardRewardSelectionScreen.cs:302-
    /// 316), so a read landing in that window sees BOTH generations and every
    /// index shifts. Empty means "no cross-check available".
    /// </summary>
    public List<string> OptionLabels = new();

    /// <summary>Screen node identity, so a stale decision can be detected.</summary>
    public GodotObject? Screen;

    /// <summary>
    /// What makes THIS decision different from the last one on the same node.
    ///
    /// Screen identity alone is not enough for the run's own screens: the map
    /// is one long-lived node for the whole act (`NMapScreen.Instance`), so
    /// the second map of act 1 is the same object as the first, and a poll
    /// keyed on identity + option count decided nothing had changed and never
    /// asked again -- map advice appeared once and never came back
    /// (2026-08-05). Overlay screens get a fresh instance each time and are
    /// covered by the default (option count + labels) either way.
    /// </summary>
    public string Signature = "";

    /// <summary>Signature, falling back to the option list when unset.</summary>
    public string SignatureOrDefault =>
        string.IsNullOrEmpty(Signature)
            ? $"{OptionCount}:{string.Join("|", OptionLabels)}"
            : Signature;

    /// <summary>
    /// True for decisions we answer ourselves instead of asking the advisor.
    /// The loot screen is the only one: "claim everything, then proceed" is
    /// not a judgement call, and round-tripping it just made F7 stall there.
    /// </summary>
    public bool Local;

    /// <summary>The advice to display/execute when <see cref="Local"/>.</summary>
    public SimAdvice? LocalAdvice;

    /// <summary>
    /// The mod's own "this is not really a decision" verdict, for what the
    /// advisor never sees (GH #874): the crystal sphere's reveals, which it
    /// answers blind. Same meaning as the advisor's `advice.hand_back`.
    /// </summary>
    public string? HandBack;

    /// <summary>Executes "pick option i". Main thread. Returns false if it
    /// could not be executed (screen gone, index invalid).</summary>
    public Func<int, bool> Choose = _ => false;

    public string ScreenName = "";

    /// <summary>
    /// The game object behind each option, in option order, where the human's
    /// pick is detected by identity rather than by a diff (SimChoiceCapture):
    /// the <c>EventOption</c> of an event button, the <c>RestSiteOption</c> of
    /// a campfire button, the <c>CardModel</c> of a deck-select holder. Empty
    /// for every other screen.
    /// </summary>
    public List<object?> OptionRefs = new();
}

internal static class SimScreenAdvice
{
    /// <summary>
    /// Describe the decision the player is being asked right now -- the map,
    /// the top of the overlay stack, or the room underneath it -- or null when
    /// there isn't one we understand. Main thread only.
    /// </summary>
    public static SimScreenDecision? Describe()
    {
        // The map outranks the overlay stack -- exactly as the game's own
        // ActiveScreenContext.GetCurrentScreen orders them (NMapScreen.
        // IsOpen is tested BEFORE NOverlayStack.Peek).
        //
        // Opening the map only HIDES the stack: NOverlayStack._Ready wires
        // NMapScreen.Opened -> HideOverlays, and screens are popped later,
        // when the next room clears them (RunManager.ClearScreens). The
        // post-combat rewards screen is never removed on Proceed either --
        // the terminal branch just calls NMapScreen.Open (RunManager.
        // ProceedFromTerminalRewardsScreen) -- so from the first fight on,
        // a spent, invisible NRewardsScreen sat on top of the stack for
        // every map choice that followed. Peeking first meant we described
        // THAT (or, once its Proceed button was disabled, nothing at all),
        // and the map advice + route overlay never came back. Events,
        // shops and rest sites push nothing, which is why they kept
        // working (2026-08-06).
        SimScreenDecision? map = Try(DescribeMap, nameof(DescribeMap));
        if (map != null)
        {
            return map;
        }

        // An open map that is not ASKING anything is not "no decision" -- it
        // is a decision already in flight, and nothing underneath it is the
        // question. Travel starts by clearing IsTravelEnabled
        // (NMapScreen.TravelToMapCoord:1054) but the map is only closed much
        // later, by RunManager.ClearScreens on the way into the next room. In
        // that window the room we are LEAVING is still in the tree, still
        // visible and still carries an enabled Proceed -- and autoplay
        // clicked it, which runs NxxxRoom.OnProceedButtonReleased ->
        // NMapScreen.Open(). That is where "the boss fight started but the map
        // is still up, I have to close it by hand" came from (godot.log
        // 2026-08-06: `NMapScreen: Boss` immediately followed by
        // `NRestSiteRoom [rest_site:proceed]`).
        if (IsMapUp())
        {
            return null;
        }

        NOverlayStack? overlays = NOverlayStack.Instance;
        IOverlayScreen? top = overlays != null && GodotObject.IsInstanceValid(overlays)
            ? SafePeek(overlays)
            : null;

        if (top is Node node && GodotObject.IsInstanceValid(node) && node.IsInsideTree())
        {
            return Try(() => top switch
            {
                NCardRewardSelectionScreen reward => DescribeCardReward(reward),
                NChooseACardSelectionScreen choose => DescribeChooseACard(choose),
                NCardGridSelectionScreen grid => DescribeCardSelect(grid),
                NRewardsScreen loot => DescribeRewards(loot),
                NChooseARelicSelection relic => DescribeRelicChoice(relic),
                NChooseABundleSelectionScreen bundle => DescribeBundle(bundle),
                NCrystalSphereScreen sphere => DescribeCrystalSphere(sphere),
                NGameOverScreen over => DescribeGameOver(over),
                _ => null,
            }, top.GetType().Name);
        }

        // An in-hand selection is checked here, below the stack and above the
        // rooms, because that is exactly when the game accepts it: clicking a
        // hand card is refused while anything is stacked
        // (`NOverlayStack.Instance.ScreenCount > 0`, NPlayerHand.
        // OnHolderPressed), so a stacked screen genuinely outranks it.
        SimScreenDecision? handSelect = Try(DescribeHandSelect, nameof(DescribeHandSelect));
        if (handSelect != null)
        {
            return handSelect;
        }

        // The shop and the rooms are not on the overlay stack -- they are
        // the run's own screens, so they are only consulted once nothing
        // is stacked on top of them. (The map is handled above: it is the
        // one run screen the game itself ranks over the stack.)
        return Try(DescribeShop, nameof(DescribeShop))
            ?? Try(DescribeEvent, nameof(DescribeEvent))
            // Below DescribeEvent, not above it: the Fake Merchant's own event
            // can still raise ordinary option buttons (it starts a fight if
            // you call out the fakes), and those outrank the shopfront.
            ?? Try(DescribeFakeMerchant, nameof(DescribeFakeMerchant))
            ?? Try(DescribeRestSite, nameof(DescribeRestSite))
            ?? Try(DescribeTreasure, nameof(DescribeTreasure));
    }

    /// <summary>
    /// Run one describer, swallowing its errors.
    ///
    /// Deliberately per-describer rather than one guard around the whole
    /// sweep: the treasure room reads `NRelic.Model`, which THROWS while the
    /// chest is still closed, and a single outer catch turned that into "this
    /// run has no decisions at all" -- autoplay sat on a shut chest for 37 s
    /// (75 consecutive `Describe failed: Model was accessed before it was
    /// set` in godot.log, 2026-08-06) until a human clicked it. One screen
    /// misbehaving must not blind the other eight.
    /// </summary>
    private static SimScreenDecision? Try(Func<SimScreenDecision?> describe, string what)
    {
        try
        {
            return describe();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimScreenAdvice] {what} failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>True while the map screen is on top of everything else.</summary>
    private static bool IsMapUp()
    {
        try
        {
            NMapScreen? map = NMapScreen.Instance;
            return map != null && GodotObject.IsInstanceValid(map) && map.IsOpen && map.IsVisibleInTree();
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>
    /// Is the room node we just found the room the RUN thinks we are in?
    ///
    /// Room nodes outlive the room: `RunManager.EnterMapPointInternal` pops
    /// the old room from the state (`ExitCurrentRooms`) and pushes the new one
    /// (`EnterRoomInternal` -> `State.PushRoom`) well before the old node
    /// leaves the tree, so a describer keyed on `IsVisibleInTree()` alone
    /// keeps offering the PREVIOUS floor's buttons. Clicking one of those is
    /// not harmless -- a stale Proceed reopens the map over the room we just
    /// entered. The run state is the authority on which room is live.
    /// </summary>
    private static bool IsCurrentRoom(RunState? runState, RoomType type)
    {
        try
        {
            return runState?.CurrentRoom?.RoomType == type;
        }
        catch (Exception)
        {
            return false;
        }
    }

    // ------------------------------------------------------------------
    // Relic offers: the boss/Ancient relic screen and the treasure chest
    // ------------------------------------------------------------------

    /// <summary>
    /// A relic holder's model, or null when the holder exists but has not been
    /// filled in yet.
    ///
    /// `NRelic.Model` is not a nullable read -- it THROWS
    /// ("Model was accessed before it was set", NRelic.cs:89) -- and the
    /// treasure room builds its holders while the chest is still shut, so the
    /// obvious `holder.Relic.Model` blew up on every poll for as long as the
    /// chest stayed closed. Read the backing field instead: absence is a
    /// normal state of this screen, not an error.
    /// </summary>
    private static readonly FieldInfo? RelicModelField =
        typeof(NRelic).GetField("_model", BindingFlags.Instance | BindingFlags.NonPublic);

    private static RelicModel? SafeRelicModel(NRelic? relic)
    {
        try
        {
            if (relic == null || !GodotObject.IsInstanceValid(relic))
            {
                return null;
            }
            return RelicModelField != null
                ? RelicModelField.GetValue(relic) as RelicModel
                : relic.Model;
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static SimScreenDecision? DescribeRelicChoice(NChooseARelicSelection screen)
    {
        List<NRelicBasicHolder> holders = UiHelper.FindAll<NRelicBasicHolder>(screen)
            .Where(h => GodotObject.IsInstanceValid(h) && h.IsEnabled && h.Visible)
            .Where(h => SafeRelicModel(h.Relic) != null)
            .ToList();
        if (holders.Count == 0)
        {
            return null;
        }

        var options = new List<Dictionary<string, object>>();
        var labels = new List<string>();
        for (int i = 0; i < holders.Count; ++i)
        {
            RelicModel relic = SafeRelicModel(holders[i].Relic)!;
            options.Add(SimRelicOptions.Option(
                i, NonCombatBridgeProtocol.PickRelicAction, relic));
            labels.Add(relic.Title.GetFormattedText());
        }

        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(runState);
        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.BossRelicState,
            ["options"] = options,
        };
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.BossRelicState,
            State = state,
            OptionCount = holders.Count,
            OptionLabels = labels,
            Screen = screen,
            ScreenName = nameof(NChooseARelicSelection),
            Choose = index => ClickHolder(holders, index),
        };
    }

    private const string TreasureRoomPath =
        "/root/Game/RootSceneContainer/Run/RoomContainer/TreasureRoom";

    /// <summary>
    /// The chest room. Only the relic pick is a judgement call; opening the
    /// chest and proceeding afterwards are answered here (Local) so autoplay
    /// walks the whole room instead of stopping on its mechanical steps.
    /// </summary>
    private static SimScreenDecision? DescribeTreasure()
    {
        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return null;
        }
        NTreasureRoom? room = tree.Root.GetNodeOrNull<NTreasureRoom>(TreasureRoomPath);
        if (room == null || !GodotObject.IsInstanceValid(room) || !room.IsVisibleInTree())
        {
            return null;
        }
        if (!IsCurrentRoom(RunManager.Instance.DebugOnlyGetState(), RoomType.Treasure))
        {
            return null;
        }

        // A holder whose relic is not filled in yet is scenery, not an offer:
        // the room builds them up front and `InitializeRelics` only runs once
        // the chest animation is done (NTreasureRoom.OpenChest).
        var holders = UiHelper.FindAll<NTreasureRoomRelicHolder>(room)
            .Where(h => GodotObject.IsInstanceValid(h) && h.IsEnabled && h.Visible)
            .Where(h => SafeRelicModel(h.Relic) != null)
            .ToList();

        if (holders.Count == 0)
        {
            // Either the chest is still closed, or the relic is taken and the
            // room is done. Both are one-button steps, not decisions.
            NProceedButton? proceed = room.ProceedButton;
            bool canProceed = proceed != null && GodotObject.IsInstanceValid(proceed) && proceed.IsEnabled;
            NClickableControl? chest = FindChest(room);
            bool canOpen = chest != null && GodotObject.IsInstanceValid(chest) &&
                           chest.IsEnabled && chest.Visible;
            // While the relics are animating in, the room re-labels Proceed as
            // SKIP and enables it 2.5 s later (OpenChest ->
            // EnableSkipAfterDelay). Treating that as the mechanical "proceed"
            // step would throw the chest's relic away, so wait for the holders
            // instead: the only Proceed we click here is a real one.
            bool isSkip = proceed != null && GodotObject.IsInstanceValid(proceed) && proceed.IsSkip;
            if (!canOpen && (!canProceed || isSkip))
            {
                return null;
            }
            string label = canOpen ? "open the chest" : "proceed";
            return LocalDecision(
                NonCombatBridgeProtocol.TreasureState, room, nameof(NTreasureRoom),
                $"treasure: {label}", label,
                index =>
                {
                    if (canOpen && chest != null && GodotObject.IsInstanceValid(chest))
                    {
                        TaskHelper.RunSafely(UiHelper.Click(chest));
                        return true;
                    }
                    if (proceed != null && GodotObject.IsInstanceValid(proceed))
                    {
                        TaskHelper.RunSafely(UiHelper.Click(proceed));
                        return true;
                    }
                    return false;
                });
        }

        if (SimMp.HasVotedOnTreasure(LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState())))
        {
            // Co-op: the relics are voted on (ties by rock-paper-scissors);
            // ours is in and the holders stay up for the others'.
            return null;
        }

        var options = new List<Dictionary<string, object>>();
        var labels = new List<string>();
        for (int i = 0; i < holders.Count; ++i)
        {
            RelicModel relic = SafeRelicModel(holders[i].Relic)!;
            options.Add(SimRelicOptions.Option(
                i, NonCombatBridgeProtocol.CollectTreasureAction, relic));
            labels.Add(relic.Title.GetFormattedText());
        }

        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(runState);
        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.TreasureState,
            // The relic-pick head is the trained surface; the chest offers
            // the same kind of choice as the boss screen, just cheaper.
            ["surface"] = "treasure",
            ["options"] = options,
        };
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.TreasureState,
            State = state,
            OptionCount = holders.Count,
            OptionLabels = labels,
            Signature = "treasure:" + string.Join("|", labels),
            Screen = room,
            ScreenName = nameof(NTreasureRoom),
            Choose = index => ClickHolder(holders, index),
        };
    }

    /// <summary>
    /// The chest button. `NTreasureRoom` binds it by unique name
    /// (`%Chest`, NTreasureRoom.cs:270) while the game's own room handler
    /// reaches it as a plain child (`Chest`); try both rather than betting the
    /// whole room on one of the two spellings.
    /// </summary>
    private static NClickableControl? FindChest(NTreasureRoom room) =>
        room.GetNodeOrNull<NClickableControl>("%Chest")
        ?? room.GetNodeOrNull<NClickableControl>("Chest");

    private static bool ClickHolder<T>(List<T> holders, int index) where T : NClickableControl
    {
        if (index < 0 || index >= holders.Count)
        {
            return false;
        }
        T holder = holders[index];
        if (!GodotObject.IsInstanceValid(holder))
        {
            return false;
        }
        TaskHelper.RunSafely(UiHelper.Click(holder));
        return true;
    }

    // ------------------------------------------------------------------
    // Rest site
    // ------------------------------------------------------------------

    private const string RestSiteRoomPath =
        "/root/Game/RootSceneContainer/Run/RoomContainer/RestSiteRoom";

    private static SimScreenDecision? DescribeRestSite()
    {
        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return null;
        }
        NRestSiteRoom? room = tree.Root.GetNodeOrNull<NRestSiteRoom>(RestSiteRoomPath);
        if (room == null || !GodotObject.IsInstanceValid(room) || !room.IsVisibleInTree())
        {
            return null;
        }
        if (!IsCurrentRoom(RunManager.Instance.DebugOnlyGetState(), RoomType.RestSite))
        {
            return null;
        }

        // An option still has to be in room.Options: after a selection the
        // game drops it from the list first and swaps the buttons later
        // (UpdateRestSiteOptions), and clicking a stale button throws (#731).
        // Co-op's Mend (heal a teammate) opens a teammate picker autoplay
        // cannot answer, and the advisor does not see teammates' HP to judge
        // it; it stays a human-only option.
        var buttons = UiHelper.FindAll<NRestSiteButton>(room)
            .Where(b => GodotObject.IsInstanceValid(b) && b.Option.IsEnabled
                && room.Options.Contains(b.Option) && b.Option.OptionId != "MEND")
            .ToList();
        if (buttons.Count == 0)
        {
            // Option already taken: what is left is the proceed click.
            NProceedButton? proceed = room.ProceedButton;
            if (proceed == null || !GodotObject.IsInstanceValid(proceed) || !proceed.IsEnabled)
            {
                return null;
            }
            return LocalDecision(
                NonCombatBridgeProtocol.RestSiteState, room, nameof(NRestSiteRoom),
                "rest site: done, proceed", "proceed",
                _ =>
                {
                    TaskHelper.RunSafely(UiHelper.Click(proceed));
                    return true;
                });
        }

        var options = new List<Dictionary<string, object>>();
        var labels = new List<string>();
        for (int i = 0; i < buttons.Count; i++)
        {
            string label = buttons[i].Option.Title.GetFormattedText();
            options.Add(new()
            {
                ["index"] = i,
                ["id"] = buttons[i].Option.OptionId,
                ["action"] = NonCombatBridgeProtocol.RestOptionAction,
                ["label"] = label,
                ["description"] = buttons[i].Option.Description.GetFormattedText(),
                ["enabled"] = true,
            });
            labels.Add(label);
        }

        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(runState);
        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.RestSiteState,
            ["options"] = options,
        };
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.RestSiteState,
            State = state,
            OptionCount = buttons.Count,
            OptionLabels = labels,
            Signature = "rest:" + string.Join("|", buttons.Select(b => b.Option.OptionId)),
            Screen = room,
            ScreenName = nameof(NRestSiteRoom),
            OptionRefs = buttons.Select(b => (object?)b.Option).ToList(),
            Choose = index => ClickHolder(buttons, index),
        };
    }

    // ------------------------------------------------------------------
    // Card bundles, the crystal sphere, and the end of the run
    // ------------------------------------------------------------------

    private static SimScreenDecision? DescribeBundle(NChooseABundleSelectionScreen screen)
    {
        List<NCardBundle> bundles = UiHelper.FindAll<NCardBundle>(screen)
            .Where(b => GodotObject.IsInstanceValid(b))
            .ToList();
        if (bundles.Count == 0)
        {
            return null;
        }

        var options = new List<Dictionary<string, object>>();
        var labels = new List<string>();
        for (int i = 0; i < bundles.Count; i++)
        {
            List<CardModel> cards = bundles[i].Bundle.ToList();
            options.Add(new()
            {
                ["index"] = i,
                ["action"] = NonCombatBridgeProtocol.PickCardBundleAction,
                ["cards"] = cards.Select(c => SimStateExtractor.SerializeCard(c)).ToList(),
                ["enabled"] = true,
            });
            labels.Add(string.Join(" + ", cards.Select(c => c.Title)));
        }

        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(runState);
        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.CardBundleState,
            ["options"] = options,
        };
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.CardBundleState,
            State = state,
            OptionCount = bundles.Count,
            OptionLabels = labels,
            Screen = screen,
            ScreenName = nameof(NChooseABundleSelectionScreen),
            Choose = index => ChooseBundle(screen, bundles, index),
        };
    }

    private static bool ChooseBundle(
        NChooseABundleSelectionScreen screen, List<NCardBundle> bundles, int index)
    {
        if (index < 0 || index >= bundles.Count || !GodotObject.IsInstanceValid(bundles[index]))
        {
            return false;
        }
        // Click the bundle, then confirm -- the screen shows a preview between
        // the two, so the confirm cannot be issued in the same frame.
        TaskHelper.RunSafely(ClickBundleAsync(screen, bundles[index]));
        return true;
    }

    private static async System.Threading.Tasks.Task ClickBundleAsync(
        NChooseABundleSelectionScreen screen, NCardBundle bundle)
    {
        await UiHelper.Click(bundle.Hitbox);
        await System.Threading.Tasks.Task.Delay(500);
        if (!GodotObject.IsInstanceValid(screen))
        {
            return;
        }
        NConfirmButton? confirm = UiHelper.FindFirst<NConfirmButton>(screen);
        if (confirm != null && GodotObject.IsInstanceValid(confirm))
        {
            await UiHelper.Click(confirm);
        }
    }

    /// <summary>
    /// The crystal sphere is a blind pick -- every remaining cell is hidden,
    /// which is the whole point of the screen -- so there is nothing for a
    /// policy to rank and no reason to spend a round trip. Answered locally:
    /// reveal the next cell, or proceed once the screen offers it.
    /// </summary>
    private static SimScreenDecision? DescribeCrystalSphere(NCrystalSphereScreen screen)
    {
        Control? cells = screen.GetNodeOrNull<Control>("%Cells");
        List<NCrystalSphereCell> hidden = cells == null
            ? new List<NCrystalSphereCell>()
            : UiHelper.FindAll<NCrystalSphereCell>(cells)
                .Where(c => GodotObject.IsInstanceValid(c) && c.Visible && c.Entity.IsHidden)
                .ToList();
        NProceedButton? proceed = screen.GetNodeOrNull<NProceedButton>("%ProceedButton");
        bool canProceed = proceed != null && GodotObject.IsInstanceValid(proceed) && proceed.IsEnabled;

        // Nothing revealed and nothing to click: the outcome is still
        // animating. That is a wait, not a decision.
        if (hidden.Count == 0 && !canProceed)
        {
            return null;
        }

        bool reveal = hidden.Count > 0 && !canProceed;
        string label = reveal ? "reveal a cell" : "proceed";
        SimScreenDecision decision = LocalDecision(
            NonCombatBridgeProtocol.CrystalSphereState, screen, nameof(NCrystalSphereScreen),
            $"crystal sphere: {label} (every cell is hidden — nothing to rank)", label,
            _ =>
            {
                if (reveal)
                {
                    return ClickHolder(hidden, 0);
                }
                if (proceed != null && GodotObject.IsInstanceValid(proceed))
                {
                    TaskHelper.RunSafely(UiHelper.Click(proceed));
                    return true;
                }
                return false;
            },
            // Every reveal is a fresh decision on the same node with the same
            // label; without the count in the signature the poll (and the
            // runaway guard) would read the second reveal as the first one
            // repeating.
            signature: $"crystal_sphere:{label}:{hidden.Count}");
        if (reveal)
        {
            // Which cell to reveal is a real choice for a player who reads the
            // board; "the first hidden cell" is not one (GH #872).
            decision.HandBack = "the AI has no policy for the Crystal Sphere yet";
        }
        return decision;
    }

    /// <summary>
    /// The run is over. Not a decision, and deliberately NOT clicked: leaving
    /// the game-over screen is the player's call, and autoplay clicking
    /// through it would bury the one screen they wanted to read. Describing it
    /// is what lets autoplay stop cleanly instead of sitting on it.
    /// </summary>
    private static SimScreenDecision? DescribeGameOver(NGameOverScreen screen) =>
        new()
        {
            AdviceFor = "game_over",
            OptionCount = 0,
            Screen = screen,
            ScreenName = nameof(NGameOverScreen),
            Local = true,
            LocalAdvice = new SimAdvice { Note = "run over — autoplay stopped" },
            Choose = _ => false,
        };

    /// <summary>A one-button decision we answer ourselves, no round trip.</summary>
    private static SimScreenDecision LocalDecision(
        string adviceFor, GodotObject screen, string screenName,
        string note, string label, Func<int, bool> choose, string? signature = null) =>
        new()
        {
            AdviceFor = adviceFor,
            OptionCount = 1,
            OptionLabels = new List<string> { label },
            Signature = signature ?? $"{adviceFor}:{label}",
            Screen = screen,
            ScreenName = screenName,
            Local = true,
            LocalAdvice = new SimAdvice
            {
                Note = note,
                Best = new SimBestAction { Action = "choose", Index = 0, Label = label },
                Options = { new SimAdviceOption { Label = label, IsBest = true } },
            },
            Choose = choose,
        };

    private static IOverlayScreen? SafePeek(NOverlayStack overlays)
    {
        try
        {
            return overlays.Peek();
        }
        catch (Exception)
        {
            // Peek throws on an empty stack in some builds; absence is normal.
            return null;
        }
    }

    // ------------------------------------------------------------------
    // Event rooms (which option to take)
    // ------------------------------------------------------------------

    private const string EventRoomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/EventRoom";

    private static SimScreenDecision? DescribeEvent()
    {
        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return null;
        }
        Node? room = tree.Root.GetNodeOrNull(EventRoomPath);
        if (room == null || !GodotObject.IsInstanceValid(room) ||
            room is CanvasItem ci && !ci.IsVisibleInTree())
        {
            return null;
        }
        if (!IsCurrentRoom(RunManager.Instance.DebugOnlyGetState(), RoomType.Event))
        {
            return null;
        }
        List<NEventOptionButton> options = UiHelper.FindAll<NEventOptionButton>(room)
            .Where(o => GodotObject.IsInstanceValid(o) && !o.Option.IsLocked)
            .ToList();
        if (options.Count == 0)
        {
            return null;
        }

        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(runState);
        if (SimMp.HasVotedOnEvent(me))
        {
            // Co-op shared event: our vote is in; the options stay up for the
            // others'. Pressing again (or the looping-event escape pressing
            // another option) would change it.
            return null;
        }
        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.EventState,
            ["options"] = SimEventRoomHandler.EventOptions(options),
        };
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.EventState,
            State = state,
            OptionCount = options.Count,
            // Same node across every event room in the run, and an event's
            // later page can offer the same NUMBER of options as its first --
            // and the same KEYS, too: a multi-page event is page after page of
            // a lone "PROCEED". The page's own description key is what
            // actually distinguishes them, and without it every page of such
            // an event looks like the first one repeating.
            Signature = $"event:{EventPageKey(runState)}:" +
                        string.Join("|", options.Select(o => o.Option.TextKey)),
            Screen = room,
            ScreenName = "NEventRoom",
            OptionRefs = options.Select(o => (object?)o.Option).ToList(),
            Choose = index => ChooseEventOption(room, index),
        };
    }

    // ------------------------------------------------------------------
    // The Fake Merchant (the one event with no event buttons)
    // ------------------------------------------------------------------

    /// <summary>
    /// `FakeMerchant` is the ONLY event in v0.107.1 with
    /// <c>EventLayoutType.Custom</c> (every other one is Normal or Combat), and
    /// its custom node puts a merchant shopfront in an event room instead of
    /// the usual option buttons. Both generic describers miss it:
    /// `DescribeShop` looks up the `MerchantRoom` node by path and this is an
    /// `EventRoom`, and `DescribeEvent` finds zero `NEventOptionButton`s.
    ///
    /// Missing it was a hard stall -- run 007 of the ascension climb sat in
    /// front of it for 180 s, and because `current_run.save` is written on
    /// entering the room, every resume walked straight back into the same
    /// event: a resume LOOP. So while the merchant is up this always returns
    /// a decision, and Leave is always enabled.
    ///
    /// Described as the engine presents it in training, an `event` with
    /// `buy`, `throw_foul`, `leave` in that order (GH #886) -- the SAME options
    /// the F7 autopilot sends (`SimEventRoomHandler.FakeMerchantOptions`), so
    /// the two lanes cannot disagree. Nothing is opened to describe it: the
    /// slots are built in `NFakeMerchant._Ready`, and the throw needs the
    /// shopfront closed anyway.
    /// </summary>
    private static SimScreenDecision? DescribeFakeMerchant()
    {
        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return null;
        }
        Node? room = tree.Root.GetNodeOrNull(EventRoomPath);
        if (room == null || !GodotObject.IsInstanceValid(room) ||
            room is CanvasItem ci && !ci.IsVisibleInTree())
        {
            return null;
        }
        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        if (!IsCurrentRoom(runState, RoomType.Event))
        {
            return null;
        }
        NFakeMerchant? merchant = UiHelper.FindFirst<NFakeMerchant>(room);
        if (merchant == null || !GodotObject.IsInstanceValid(merchant) ||
            !merchant.IsVisibleInTree())
        {
            return null;
        }

        Player? me = LocalContext.GetMe(runState);
        NMerchantInventory inventory = merchant.Inventory;
        if (inventory == null || !GodotObject.IsInstanceValid(inventory))
        {
            return FakeMerchantLeaveDecision(merchant, "fake merchant: no shopfront — leave");
        }

        NMerchantSlot? buySlot = SimEventRoomHandler.FakeMerchantBuySlot(merchant);
        // The game's own usability check fails while the shopfront is open
        // (`FoulPotion.GetFoulPotionMerchantTarget`), and a player may have
        // opened it. Holding the potion is what decides the option; the
        // throw closes the shopfront first.
        bool canThrow = SimEventRoomHandler.FakeMerchantFoulPotion(merchant, me) != null ||
                        inventory.IsOpen && merchant.MerchantButton is { } button &&
                        button.IsVisibleInTree() && me != null && me.Potions.OfType<FoulPotion>().Any();
        List<Dictionary<string, object?>> options = SimEventRoomHandler.FakeMerchantOptions(buySlot, canThrow);

        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.EventState,
            ["options"] = options,
            ["player_gold"] = me?.Gold ?? 0,
            ["stock"] = inventory.GetAllSlots().Select(SimShopRoomHandler.ShopStockRow).ToList(),
        };
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.EventState,
            State = state,
            OptionCount = options.Count,
            OptionLabels = options.Select(o => (string)o["label"]!).ToList(),
            // A purchase changes the gold and the next relic, and may take
            // the buy or the throw away; each is a new decision.
            Signature = $"fakemerchant:{me?.Gold ?? 0}:{options[SimEventRoomHandler.FakeMerchantBuy]["relic_id"]}:" +
                        string.Join("|", options.Select(o => o["enabled"] is true ? "1" : "0")),
            Screen = merchant,
            ScreenName = nameof(NFakeMerchant),
            Choose = index => ChooseFakeMerchantOption(merchant, room, options, buySlot, me, index),
        };
    }

    private static SimScreenDecision FakeMerchantLeaveDecision(NFakeMerchant merchant, string label) =>
        LocalDecision(
            NonCombatBridgeProtocol.EventState, merchant, nameof(NFakeMerchant),
            label, "leave",
            _ =>
            {
                TaskHelper.RunSafely(LeaveFakeMerchantAsync(merchant));
                return true;
            });

    private static bool ChooseFakeMerchantOption(
        NFakeMerchant merchant, Node room, List<Dictionary<string, object?>> options,
        NMerchantSlot? buySlot, Player? me, int index)
    {
        if (!GodotObject.IsInstanceValid(merchant) || index < 0 || index >= options.Count ||
            options[index]["enabled"] is not true)
        {
            return false;
        }
        switch (index)
        {
            case SimEventRoomHandler.FakeMerchantBuy:
                TaskHelper.RunSafely(SimEventRoomHandler.BuyFromFakeMerchantAsync(
                    merchant, buySlot!, CancellationToken.None));
                return true;
            case SimEventRoomHandler.FakeMerchantThrowFoul:
                TaskHelper.RunSafely(ThrowAtFakeMerchantAsync(merchant, room, me));
                return true;
            default:
                TaskHelper.RunSafely(LeaveFakeMerchantAsync(merchant));
                return true;
        }
    }

    /// <summary>
    /// Close the shopfront if a player left it open (the potion cannot target
    /// the merchant otherwise), then throw. The fight that follows is a
    /// normal combat screen for the coach.
    /// </summary>
    private static async System.Threading.Tasks.Task ThrowAtFakeMerchantAsync(
        NFakeMerchant merchant, Node room, Player? me)
    {
        NMerchantInventory inventory = merchant.Inventory;
        if (inventory.IsOpen)
        {
            NBackButton? back = UiHelper.FindFirst<NBackButton>(merchant);
            if (back != null && GodotObject.IsInstanceValid(back) && back.IsVisibleInTree())
            {
                await UiHelper.Click(back);
            }
            await WaitHelper.Until(() => !inventory.IsOpen, CancellationToken.None,
                TimeSpan.FromSeconds(10), "FakeMerchant shopfront did not close");
        }
        FoulPotion? foul = SimEventRoomHandler.FakeMerchantFoulPotion(merchant, me);
        if (foul == null)
        {
            Logger.Log("[SimScreenAdvice] No throwable Foul Potion after all; asking again.");
            return;
        }
        await SimEventRoomHandler.ThrowFoulPotionAsync(foul, room, CancellationToken.None);
    }

    /// <summary>
    /// Close the shopfront, then Proceed -- the same two-step the real shop
    /// needs, for the same reason: `OpenInventory` disables the Proceed button
    /// until the inventory reports itself closed (NFakeMerchant.cs:338-352).
    /// Proceed is wired to `HideScreen`, which opens the map
    /// (NFakeMerchant.cs:328), so this is what ends the event.
    /// </summary>
    private static async System.Threading.Tasks.Task LeaveFakeMerchantAsync(NFakeMerchant merchant)
    {
        NBackButton? back = UiHelper.FindFirst<NBackButton>(merchant);
        if (back != null && GodotObject.IsInstanceValid(back) && back.IsVisibleInTree())
        {
            await UiHelper.Click(back);
            await System.Threading.Tasks.Task.Delay(300);
        }
        if (!GodotObject.IsInstanceValid(merchant))
        {
            return;
        }
        NProceedButton? proceed = merchant.GetNodeOrNull<NProceedButton>("%ProceedButton");
        if (proceed != null && GodotObject.IsInstanceValid(proceed))
        {
            await UiHelper.Click(proceed);
        }
    }

    /// <summary>
    /// Which page of the event we are on, as its description's loc key.
    /// Empty when the room is between pages or is not an event after all.
    /// </summary>
    private static string EventPageKey(RunState? runState)
    {
        try
        {
            if (runState?.CurrentRoom is EventRoom room)
            {
                return room.LocalMutableEvent.Description?.LocEntryKey ?? "";
            }
        }
        catch (Exception)
        {
            // Between pages the event synchronizer has no local event yet.
        }
        return "";
    }

    private static bool ChooseEventOption(Node room, int index)
    {
        if (!GodotObject.IsInstanceValid(room))
        {
            return false;
        }
        List<NEventOptionButton> options = UiHelper.FindAll<NEventOptionButton>(room)
            .Where(o => GodotObject.IsInstanceValid(o) && !o.Option.IsLocked)
            .ToList();
        if (index < 0 || index >= options.Count)
        {
            return false;
        }
        // A click after the event moved on (into a spawned combat) NREs
        // inside the button's own handler -- the autopilot hit this on seed
        // 1007 and lost the run to it. Same guard here.
        if (NEventRoom.Instance == null)
        {
            Logger.Log("[SimScreenAdvice] Event room is transitioning; not clicking.");
            return false;
        }
        TaskHelper.RunSafely(UiHelper.Click(options[index]));
        return true;
    }

    // ------------------------------------------------------------------
    // Shop (what to buy with the gold you have)
    // ------------------------------------------------------------------

    private const string MerchantRoomPath = "/root/Game/RootSceneContainer/Run/RoomContainer/MerchantRoom";

    private static SimScreenDecision? DescribeShop()
    {
        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return null;
        }
        NMerchantRoom? room = tree.Root.GetNodeOrNull<NMerchantRoom>(MerchantRoomPath);
        if (room == null || !GodotObject.IsInstanceValid(room) || !room.IsVisibleInTree())
        {
            return null;
        }
        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        if (runState == null || !IsCurrentRoom(runState, RoomType.Shop))
        {
            return null;
        }
        // Leaving takes two clicks 300 ms apart (close the shopfront, then
        // proceed), and the 500 ms poll lands inside that window: it saw a
        // closed inventory, re-opened it, and the Proceed we were about to
        // click is disabled while the inventory is open. Observed live
        // 2026-08-06 -- harmless that time because the proceed still landed,
        // but it is a loop waiting to happen. While a leave is in flight the
        // shop is not a decision.
        if (ReferenceEquals(_leavingShop, room))
        {
            return null;
        }

        Player? me = LocalContext.GetMe(runState);

        // Show the stock ONCE per visit. Purchases go straight through the
        // merchant entry (the same call the autopilot handler makes), which
        // needs no UI at all -- so in advisor mode the shop was bought out
        // from behind a closed shopfront and the human watched the AI stare
        // at the merchant (reported 2026-08-06). But re-opening on every
        // poll trapped the human IN the shopfront: their back arrow closed
        // it and the next 500 ms poll shoved it open again (reported
        // 2026-08-21). One forced open per room instance; after that the
        // human owns the shopfront, and slot enumeration below works against
        // a closed inventory anyway.
        if (!room.Inventory.IsOpen && !ReferenceEquals(_shopfrontOpenedFor, room))
        {
            Logger.Log("[SimScreenAdvice] Opening the merchant inventory (once per visit).");
            _shopfrontOpenedFor = room;
            room.OpenInventory();
        }

        // Same legality filter as the autopilot handler, including the
        // potion gate (full belt, Sozu: an unbuyable potion would otherwise
        // be offered as advice the player cannot act on).
        List<NMerchantSlot> slots = room.Inventory.GetAllSlots()
            .Where(slot => slot.Entry.IsStocked && slot.Entry.EnoughGold
                && !(me != null && slot.Entry is MerchantPotionEntry p && !SimPotionRules.CanProcure(me, p.Model)))
            .ToList();
        if (slots.Count == 0)
        {
            // Nothing affordable left. There is no judgement here, but there
            // IS still a step -- and returning null used to leave autoplay
            // parked in front of a merchant it could not buy from, now with an
            // open shopfront in the way.
            return LocalDecision(
                NonCombatBridgeProtocol.ShopState, room, nameof(NMerchantRoom),
                "shop: nothing left to buy — leave", "leave shop",
                _ =>
                {
                    TaskHelper.RunSafely(LeaveShopAsync(room));
                    return true;
                });
        }

        var options = new List<Dictionary<string, object>>
        {
            new()
            {
                ["index"] = 0,
                ["id"] = NonCombatBridgeProtocol.LeaveShopAction,
                ["action"] = NonCombatBridgeProtocol.LeaveShopAction,
                ["label"] = "Leave shop",
                ["enabled"] = true,
            },
        };
        options.AddRange(slots.Select((slot, i) => SimShopRoomHandler.ShopOption(slot, i + 1)));

        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.ShopState,
            ["options"] = options,
            ["player_gold"] = me?.Gold ?? 0,
        };
        int? purgeCost = SimShopRoomHandler.SafePurgeCost(slots);
        if (purgeCost.HasValue)
        {
            state["purge_cost"] = purgeCost.Value;
        }
        // The WHOLE shopfront (sold-out and unaffordable slots included) for
        // the workbench mirror; `options` above stays purchasable-only
        // because it is what the policy ranks.
        try
        {
            state["stock"] = room.Inventory.GetAllSlots()
                .Select(SimShopRoomHandler.ShopStockRow)
                .ToList();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimScreenAdvice] shop stock serialization failed: {ex.Message}");
        }
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.ShopState,
            State = state,
            OptionCount = options.Count,
            // Gold too: buying does not always shrink the stock (a purge
            // leaves the slot count alone), but it always spends.
            Signature = $"shop:{me?.Gold ?? 0}:" +
                        string.Join("|", options.Select(o => o.TryGetValue("id", out object? id) ? id : "?")),
            Screen = room,
            ScreenName = nameof(NMerchantRoom),
            Choose = index => ChooseShopOption(room, slots, index),
        };
    }

    /// <summary>
    /// Index 0 is "leave shop" (click proceed); 1..n buy the corresponding
    /// slot through the game's own purchase entry point, the same one the
    /// autopilot handler calls. Anything the purchase opens (Dolly's Mirror's
    /// card select, Orrery's rewards) simply becomes the next decision the
    /// advisor describes, so no screen-drain pump is needed here.
    /// </summary>
    private static bool ChooseShopOption(NMerchantRoom room, List<NMerchantSlot> slots, int index)
    {
        if (!GodotObject.IsInstanceValid(room))
        {
            return false;
        }
        if (index <= 0)
        {
            if (!GodotObject.IsInstanceValid(room.ProceedButton))
            {
                return false;
            }
            TaskHelper.RunSafely(LeaveShopAsync(room));
            return true;
        }
        int slotIndex = index - 1;
        if (slotIndex >= slots.Count)
        {
            return false;
        }
        NMerchantSlot slot = slots[slotIndex];
        if (!slot.Entry.IsStocked || !slot.Entry.EnoughGold)
        {
            return false;
        }
        TaskHelper.RunSafely(slot.Entry.OnTryPurchaseWrapper(room.Inventory.Inventory));
        return true;
    }

    /// <summary>
    /// Shut the shopfront, then proceed -- the order the game's own handler
    /// uses. `OpenInventory` disables the room's Proceed button until the
    /// inventory reports itself closed (NMerchantRoom.cs:302-312), so leaving
    /// straight from an open inventory would force a click on a button the
    /// room considers unavailable.
    /// </summary>
    private static async System.Threading.Tasks.Task LeaveShopAsync(NMerchantRoom room)
    {
        // Claim the in-flight flag HERE, not at the call sites. It was set by
        // the "Leave shop" option only, and both live exits went through the
        // other door -- the "nothing left to buy" local decision -- so the
        // shopfront reopened anyway (2026-08-06). One owner, no caller can
        // forget. Runs synchronously: RunSafely starts the task inline and
        // there is no await above this line.
        _leavingShop = room;
        try
        {
            NBackButton? back = UiHelper.FindFirst<NBackButton>(room);
            if (back != null && GodotObject.IsInstanceValid(back) && back.IsVisibleInTree())
            {
                await UiHelper.Click(back);
                await System.Threading.Tasks.Task.Delay(300);
            }
            if (GodotObject.IsInstanceValid(room) && GodotObject.IsInstanceValid(room.ProceedButton))
            {
                await UiHelper.Click(room.ProceedButton);
            }
        }
        finally
        {
            // Always released, including on failure: the shop then becomes a
            // decision again and the leave is simply retried, rather than the
            // room going permanently undescribed.
            if (ReferenceEquals(_leavingShop, room))
            {
                _leavingShop = null;
            }
        }
    }

    /// <summary>The shop whose two-click exit is currently in flight.</summary>
    private static NMerchantRoom? _leavingShop;

    /// <summary>The shop room whose shopfront we already force-opened once;
    /// afterwards the human may close it without us re-opening it.</summary>
    private static NMerchantRoom? _shopfrontOpenedFor;

    // ------------------------------------------------------------------
    // Map (which node to walk to next)
    // ------------------------------------------------------------------

    /// <summary>
    /// Advisor-mode twin of the autopilot's map_select: the reachable next
    /// nodes, offered only while the map is open AND travellable (it can be
    /// peeked at from inside a room, where nothing can be walked to).
    /// </summary>
    private static SimScreenDecision? DescribeMap()
    {
        NMapScreen? map = NMapScreen.Instance;
        if (map == null || !GodotObject.IsInstanceValid(map) || !map.IsOpen ||
            !map.IsVisibleInTree() || !map.IsTravelEnabled)
        {
            return null;
        }
        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        if (runState == null || SimMp.HasVotedOnMap(LocalContext.GetMe(runState)))
        {
            // Co-op: our vote is in; the map stays open for the others'.
            return null;
        }
        List<NMapPoint> available = SimMapHandler.AvailableNodes(map, runState);
        if (available.Count == 0)
        {
            return null;
        }

        // Same node shape as the autopilot's map_select.
        List<Dictionary<string, object>> nodes = available.Select((p, i) => new Dictionary<string, object>
        {
            ["index"] = i,
            ["type"] = p.Point.PointType.ToString(),
            ["row"] = p.Point.coord.row,
            ["col"] = p.Point.coord.col,
        }).ToList();

        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = "map_select",
            ["nodes"] = nodes,
            ["map"] = SimStateExtractor.SerializeMapGraph(runState),
        };
        SimStateExtractor.ApplyEnvelope(state, runState, LocalContext.GetMe(runState));

        // The map node lives for the whole act, so its identity cannot tell
        // two map decisions apart; the visit count plus the offer can.
        string offered = string.Join("|", available.Select(p => $"{p.Point.coord.col},{p.Point.coord.row}"));
        return new SimScreenDecision
        {
            AdviceFor = "map_select",
            State = state,
            OptionCount = available.Count,
            Signature = "map:" + runState.VisitedMapCoords.Count + ":" + offered,
            Screen = map,
            ScreenName = nameof(NMapScreen),
            Choose = ChooseMapNode,
        };
    }

    private static bool ChooseMapNode(int index)
    {
        NMapScreen? map = NMapScreen.Instance;
        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        if (map == null || !GodotObject.IsInstanceValid(map) || runState == null)
        {
            return false;
        }
        List<NMapPoint> available = SimMapHandler.AvailableNodes(map, runState);
        if (index < 0 || index >= available.Count)
        {
            return false;
        }
        NMapPoint point = available[index];
        if (!GodotObject.IsInstanceValid(point))
        {
            Logger.Log($"[SimScreenAdvice] Map node {index} is gone.");
            return false;
        }
        // Map points are briefly disabled while the screen animates in --
        // the autopilot handler waits for IsEnabled before clicking, and
        // refusing outright here made F6 look like it did nothing.
        TaskHelper.RunSafely(ClickMapPointAsync(point));
        return true;
    }

    private static async System.Threading.Tasks.Task ClickMapPointAsync(NMapPoint point)
    {
        for (int attempt = 0; attempt < 20; attempt++)
        {
            if (!GodotObject.IsInstanceValid(point))
            {
                return;
            }
            if (point.IsEnabled)
            {
                await UiHelper.Click(point);
                return;
            }
            await System.Threading.Tasks.Task.Delay(100);
        }
        Logger.Log("[SimScreenAdvice] Map node never became clickable.");
    }

    // ------------------------------------------------------------------
    // Card reward (post-combat pick)
    // ------------------------------------------------------------------

    private static SimScreenDecision? DescribeCardReward(NCardRewardSelectionScreen screen)
    {
        List<NGridCardHolder> holders = RewardHolders(screen);
        if (holders.Count == 0)
        {
            return null;
        }

        var cards = new List<Dictionary<string, object?>>();
        var labels = new List<string>();
        for (int i = 0; i < holders.Count; ++i)
        {
            CardModel? card = holders[i].CardModel;
            Dictionary<string, object?> data = card != null
                ? SimStateExtractor.SerializeCard(card)
                : new Dictionary<string, object?>();
            data["index"] = i;
            cards.Add(data);
            labels.Add(card?.Title ?? "");
        }

        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(runState);
        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.CardRewardState,
            ["cards"] = cards,
            ["can_skip"] = true,
        };
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.CardRewardState,
            State = state,
            OptionCount = holders.Count,
            OptionLabels = labels,
            Screen = screen,
            ScreenName = nameof(NCardRewardSelectionScreen),
            Choose = index => ChooseCardReward(screen, index),
        };
    }

    /// <summary>
    /// The offered cards, in screen order. Deliberately NOT
    /// `UiHelper.FindAll&lt;NCardHolder&gt;`: that walks the whole subtree in
    /// traversal order and also matches holders that are not offers (the
    /// reward-alternative buttons under `UI/RewardAlternatives`, previews),
    /// which shifted every index — F6 took the wrong card (2026-08-05). The
    /// screen's own accessors read the children of `_cardRow` and nothing
    /// else (NCardRewardSelectionScreen.cs:223/302/366), so we do the same.
    /// </summary>
    private static List<NGridCardHolder> RewardHolders(NCardRewardSelectionScreen screen) =>
        CardRowHolders(screen, typeof(NCardRewardSelectionScreen), fallbackScreen: screen);

    /// <summary>
    /// The `_cardRow` children of any screen that lays its offer out that way
    /// (the post-combat reward and the in-combat "choose a card" both do).
    /// </summary>
    private static List<NGridCardHolder> CardRowHolders(
        GodotObject screen, Type screenType, Node? fallbackScreen)
    {
        try
        {
            FieldInfo? field = screenType.GetField(
                "_cardRow", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field?.GetValue(screen) is Node row && GodotObject.IsInstanceValid(row))
            {
                // QueueFree'd nodes stay children until the end of the frame,
                // and `SetOptions` frees the previous offer and adds the new
                // one in a single call -- read in that window, the row holds
                // both generations. Dropping the dying ones (and the
                // model-less ones, which are never a clickable offer) keeps
                // index i meaning "the i-th card the player can see".
                return row.GetChildren(false).OfType<NGridCardHolder>()
                    .Where(h => GodotObject.IsInstanceValid(h) &&
                                !h.IsQueuedForDeletion() &&
                                h.CardModel != null)
                    .ToList();
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimScreenAdvice] Card row unavailable: {ex.Message}");
        }
        // Fallback: the grid holders anywhere under the screen, which at
        // least excludes the alternative buttons (they are NButtons).
        if (fallbackScreen == null || !GodotObject.IsInstanceValid(fallbackScreen))
        {
            return new List<NGridCardHolder>();
        }
        return UiHelper.FindAll<NGridCardHolder>(fallbackScreen)
            .Where(h => GodotObject.IsInstanceValid(h) && h.CardModel != null)
            .ToList();
    }

    // ------------------------------------------------------------------
    // "Choose a card" (Skill/Attack/Power potions, Discovery-style effects)
    // ------------------------------------------------------------------

    /// <summary>
    /// The in-combat card generator (`CardSelectCmd.FromChooseACardScreen`,
    /// used by SkillPotion and friends): pick 1 of 3 freshly generated cards,
    /// usually skippable.
    ///
    /// Autopilot never sees this screen at all -- with an `ICardSelector`
    /// installed, `FromChooseACardScreen` answers the callback and returns
    /// before any UI exists (CardSelectCmd.cs:228-231). In advisor mode there
    /// is no selector, so the real screen opens, and until now nothing
    /// described it: no advice, and F7 sat on it forever (2026-08-05).
    /// </summary>
    private static SimScreenDecision? DescribeChooseACard(NChooseACardSelectionScreen screen)
    {
        List<NGridCardHolder> holders =
            CardRowHolders(screen, typeof(NChooseACardSelectionScreen), fallbackScreen: screen);
        if (holders.Count == 0)
        {
            return null;
        }

        var cards = new List<Dictionary<string, object?>>();
        var labels = new List<string>();
        for (int i = 0; i < holders.Count; ++i)
        {
            CardModel card = holders[i].CardModel!;
            Dictionary<string, object?> data = SimStateExtractor.SerializeCard(card);
            data["index"] = i;
            cards.Add(data);
            labels.Add(card.Title);
        }

        bool canSkip = SkipButton(screen) != null;
        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(runState);
        Dictionary<string, object> state = new()
        {
            // The card-pick head is the closest trained surface, so it rides
            // the card_reward dialect -- but flagged, because the meaning is
            // not the same: this copy is generated INTO THE FIGHT, not added
            // to the deck, so a "bad deck card" can still be the right pick.
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.CardRewardState,
            ["surface"] = "choose_a_card",
            ["cards"] = cards,
            ["can_skip"] = canSkip,
            ["combat"] = (object?)SimCombatHandler.BuildEmbeddedCombatState(me)!,
        };
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.CardRewardState,
            State = state,
            OptionCount = holders.Count,
            OptionLabels = labels,
            Screen = screen,
            ScreenName = nameof(NChooseACardSelectionScreen),
            Choose = index => ChooseGeneratedCard(screen, index),
        };
    }

    /// <summary>The skip button, or null when this choice cannot be skipped.</summary>
    private static NChoiceSelectionSkipButton? SkipButton(NChooseACardSelectionScreen screen)
    {
        try
        {
            FieldInfo? field = typeof(NChooseACardSelectionScreen).GetField(
                "_skipButton", BindingFlags.Instance | BindingFlags.NonPublic);
            // The screen hides AND disables the button when canSkip is false
            // (NChooseACardSelectionScreen.cs:322-323), so visibility is the
            // honest test for "is skipping on the table".
            if (field?.GetValue(screen) is NChoiceSelectionSkipButton button &&
                GodotObject.IsInstanceValid(button) && button.IsVisibleInTree())
            {
                return button;
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimScreenAdvice] Skip button unavailable: {ex.Message}");
        }
        return null;
    }

    private static bool ChooseGeneratedCard(NChooseACardSelectionScreen screen, int index)
    {
        if (!(GodotObject.IsInstanceValid(screen) && screen.IsVisibleInTree()))
        {
            return false;
        }
        if (index < 0)
        {
            return false;
        }
        List<NGridCardHolder> holders =
            CardRowHolders(screen, typeof(NChooseACardSelectionScreen), fallbackScreen: screen);
        if (index >= holders.Count)
        {
            NChoiceSelectionSkipButton? skip = SkipButton(screen);
            if (skip == null)
            {
                Logger.Log("[SimScreenAdvice] Choose-a-card: skip recommended but not offered.");
                return false;
            }
            TaskHelper.RunSafely(UiHelper.Click(skip));
            return true;
        }
        NGridCardHolder holder = holders[index];
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
        return true;
    }

    private static bool ChooseCardReward(NCardRewardSelectionScreen screen, int index)
    {
        if (!(GodotObject.IsInstanceValid(screen) && screen.IsVisibleInTree()))
        {
            return false;
        }
        List<NGridCardHolder> holders = RewardHolders(screen);
        if (index < 0)
        {
            return false;
        }
        // Past the last card == "skip the reward", the same spelling the
        // autopilot dialect uses (an out-of-range choose is a skip).
        if (index >= holders.Count)
        {
            return TrySkipCardReward(screen);
        }
        NGridCardHolder holder = holders[index];
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
        return true;
    }

    /// <summary>
    /// Skip a card reward by completing the screen's private
    /// `_completionSource` with null -- the same mechanism
    /// SimCardRewardScreenHandler uses (CardReward.cs:260-267 treats a null
    /// result as end-selection-without-reward). There is no skip button to
    /// click.
    /// </summary>
    private static bool TrySkipCardReward(NCardRewardSelectionScreen screen)
    {
        try
        {
            FieldInfo? field = typeof(NCardRewardSelectionScreen).GetField(
                "_completionSource", BindingFlags.Instance | BindingFlags.NonPublic);
            var tcs = (System.Threading.Tasks.TaskCompletionSource<int?>?)field?.GetValue(screen);
            if (tcs == null || tcs.Task.IsCompleted)
            {
                return false;
            }
            tcs.SetResult(null);
            // The reward survives the skip (see SkippedCardRewards); record
            // the decision or the loot screen will offer it straight back.
            if (_openedCardReward != null)
            {
                SkippedCardRewards.Add(_openedCardReward);
                _openedCardReward = null;
            }
            return true;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimScreenAdvice] Card-reward skip failed: {ex.Message}");
            return false;
        }
    }

    // ------------------------------------------------------------------
    // Post-combat loot screen (answered locally: take it all, then proceed)
    // ------------------------------------------------------------------

    private static SimScreenDecision? DescribeRewards(NRewardsScreen screen)
    {
        List<NRewardButton> buttons = ClaimableRewards(screen);
        NProceedButton? proceed = UiHelper.FindFirst<NProceedButton>(screen);
        if (buttons.Count == 0 && (proceed == null || !proceed.IsEnabled))
        {
            return null;
        }

        var advice = new SimAdvice { Note = "loot: claim everything, then proceed" };
        for (int i = 0; i < buttons.Count; i++)
        {
            advice.Options.Add(new SimAdviceOption
            {
                Label = RewardLabel(buttons[i]),
                IsBest = i == 0,
            });
        }
        advice.Options.Add(new SimAdviceOption
        {
            Label = "proceed",
            IsBest = buttons.Count == 0,
        });
        int bestIndex = buttons.Count == 0 ? buttons.Count : 0;
        advice.Best = new SimBestAction
        {
            Action = "choose",
            Index = bestIndex,
            Label = advice.Options[bestIndex].Label,
        };

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.RewardScreenState,
            OptionCount = buttons.Count + 1,
            Screen = screen,
            ScreenName = nameof(NRewardsScreen),
            Local = true,
            LocalAdvice = advice,
            Choose = index => ChooseReward(screen, index),
        };
    }

    /// <summary>
    /// Card rewards whose screen we opened and then skipped.
    ///
    /// Skipping a card reward does NOT retire it, by design: the game's Skip
    /// button is `EndSelectionAndDoNotCompleteReward`, documented as "end card
    /// selection, but don't complete it -- the player may re-enter card
    /// selection" (PostAlternateCardRewardAction). `CardReward.OnSelect` duly
    /// returns false and the loot screen re-enables the button.
    ///
    /// A human sees an offer they already declined. Autoplay saw an unclaimed
    /// reward, claimed it, skipped it, claimed it -- forever, at two decisions
    /// a second, with the run frozen on floor 5 (2026-08-06). Note the two
    /// choices ALTERNATE, so "the same choice repeated" never catches this.
    /// Remembering the decision is the fix: we declined that reward, so the
    /// only move left on the loot screen is Proceed.
    ///
    /// Reference identity, and never cleared: rewards are freshly constructed
    /// per fight, so a stale entry can never match a later one.
    /// </summary>
    private static readonly HashSet<object> SkippedCardRewards = new(ReferenceEqualityComparer.Instance);

    /// <summary>The card reward whose screen we most recently opened, so a
    /// skip on that screen can be attributed back to it.</summary>
    private static object? _openedCardReward;

    /// <summary>
    /// Reward buttons that can be claimed right now: live and enabled,
    /// potions only when the player can take them (free slot, no Sozu), and
    /// never a card reward we already opened and skipped.
    /// </summary>
    private static List<NRewardButton> ClaimableRewards(NRewardsScreen screen)
    {
        Player? me = LocalContext.GetMe(RunManager.Instance.DebugOnlyGetState());
        return UiHelper.FindAll<NRewardButton>(screen)
            .Where(b => GodotObject.IsInstanceValid(b) && b.IsEnabled)
            .Where(b => b.Reward is not PotionReward pr || (me != null && SimPotionRules.CanProcure(me, pr.Potion)))
            .Where(b => b.Reward == null || !SkippedCardRewards.Contains(b.Reward))
            .ToList();
    }

    /// <summary>Advisor-facing label (note "gold: N" with a space; not the wire id).</summary>
    private static string RewardLabel(NRewardButton button)
    {
        try
        {
            return button.Reward switch
            {
                null => "unknown",
                GoldReward gold => $"gold: {gold.Amount}",
                Reward reward => reward.GetType().Name,
            };
        }
        catch (Exception)
        {
            return "unknown";
        }
    }

    private static bool ChooseReward(NRewardsScreen screen, int index)
    {
        if (!GodotObject.IsInstanceValid(screen))
        {
            return false;
        }
        List<NRewardButton> buttons = ClaimableRewards(screen);
        if (index >= 0 && index < buttons.Count)
        {
            // Remember which card reward this opens, so that if its screen is
            // skipped we know not to offer the button again.
            _openedCardReward = buttons[index].Reward is CardReward card ? card : null;
            TaskHelper.RunSafely(UiHelper.Click(buttons[index]));
            return true;
        }
        NProceedButton? proceed = UiHelper.FindFirst<NProceedButton>(screen);
        if (proceed == null || !GodotObject.IsInstanceValid(proceed) || !proceed.IsEnabled)
        {
            return false;
        }
        TaskHelper.RunSafely(UiHelper.Click(proceed));
        return true;
    }

    // ------------------------------------------------------------------
    // Card select (Armaments, rest-site smith, transform, enchant, grants)
    // ------------------------------------------------------------------

    /// <summary>
    /// How many cards this screen already has selected, or -1 when the screen
    /// does not keep a `_selectedCards` set we recognise.
    ///
    /// Every deck-select screen in v0.107.1 spells it the same way
    /// (NSimpleCardSelectScreen, NDeckCardSelectScreen, NDeckUpgradeSelect-
    /// Screen, NDeckEnchantSelectScreen, NDeckTransformSelectScreen,
    /// NCombatPileCardSelectScreen), and it is the only honest way to tell
    /// "nothing picked yet" from "picked, waiting to be confirmed" -- an
    /// enabled Confirm button does NOT mean a pick has happened
    /// (NSimpleCardSelectScreen._Ready enables it up front whenever
    /// MinSelect == 0).
    /// </summary>
    private static int SelectedCount(NCardGridSelectionScreen screen)
        => SelectedCards(screen)?.Count ?? -1;

    /// <summary>
    /// The cards this screen already has selected, or null when the screen
    /// does not keep a `_selectedCards` set we recognise. See
    /// <see cref="SelectedCount" /> for why this field is the honest source.
    ///
    /// The identities matter, not just the count: `_selectedCards` is a
    /// HashSet, so clicking an already-selected card **de-selects** it
    /// (NDeckCardSelectScreen.cs:262). Offering a picked card again is
    /// therefore a way to loop forever on a screen that needs two cards.
    /// </summary>
    private static HashSet<CardModel>? SelectedCards(NCardGridSelectionScreen screen)
    {
        try
        {
            FieldInfo? field = screen.GetType().GetField(
                "_selectedCards", BindingFlags.Instance | BindingFlags.NonPublic);
            // HashSet&lt;T&gt; does not implement the non-generic ICollection,
            // so walk it the one way that works for any set or list shape.
            if (field?.GetValue(screen) is System.Collections.IEnumerable selected)
            {
                var cards = new HashSet<CardModel>();
                foreach (object? item in selected)
                {
                    if (item is CardModel card)
                    {
                        cards.Add(card);
                    }
                }
                return cards;
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimScreenAdvice] Selection count unavailable: {ex.Message}");
        }
        return null;
    }

    /// <summary>
    /// The holders this screen will still accept a pick on: everything on the
    /// grid minus what is already selected. Both the description and the click
    /// go through here so an index means the same thing on both sides.
    /// </summary>
    private static List<NGridCardHolder> SelectableHolders(NCardGridSelectionScreen screen)
    {
        NCardGrid? grid = UiHelper.FindFirst<NCardGrid>(screen);
        if (grid == null || !GodotObject.IsInstanceValid(grid))
        {
            return new List<NGridCardHolder>();
        }
        HashSet<CardModel>? selected = SelectedCards(screen);
        return grid.CurrentlyDisplayedCardHolders
            .Where(h => GodotObject.IsInstanceValid(h) && h.CardModel != null)
            .Where(h => selected == null || !selected.Contains(h.CardModel!))
            .ToList();
    }

    /// <summary>
    /// The confirm button this screen is currently waiting on, or null.
    ///
    /// These screens do NOT share one confirm button, and that is what made
    /// F7 stall on every smith and every enchant and let the shop's card
    /// removal collect a growing pile of Strikes (reported 2026-08-06):
    ///
    ///   NSimpleCardSelectScreen    %Confirm completes the selection.
    ///   NDeckCardSelectScreen      %Confirm only OPENS a preview; the
    ///     / NDeckTransformSelect   selection is completed by the preview
    ///                              container's own confirm -- and with
    ///                              MinSelect == MaxSelect (a shop purge)
    ///                              %Confirm never enables at ALL, because
    ///                              picking the last card opens the preview
    ///                              directly (OnCardClicked -> PreviewSelection).
    ///   NDeckUpgradeSelectScreen   no main confirm; clicking a card shows
    ///                              %Upgrade{Single,Multi}PreviewContainer
    ///                              whose child `Confirm` completes it.
    ///   NDeckEnchantSelectScreen   `Confirm` (NOT `%Confirm`), then
    ///                              %Enchant{Single,Multi}PreviewContainer/Confirm.
    ///
    /// Looking only for `%Confirm` therefore found a disabled button, or no
    /// button, on three of the four -- so the pick was never committed, the
    /// screen stayed open, and the next poll picked another card. Ask instead
    /// for "an enabled, visible confirm anywhere on this screen", preferring
    /// one inside a preview container: while a preview is up the main confirm
    /// can still be live, and re-pressing it just re-opens the preview.
    /// </summary>
    private static NConfirmButton? PendingConfirm(NCardGridSelectionScreen screen)
    {
        List<NConfirmButton> buttons = UiHelper.FindAll<NConfirmButton>(screen)
            .Where(b => GodotObject.IsInstanceValid(b) && b.IsEnabled && b.IsVisibleInTree())
            .ToList();
        if (buttons.Count == 0)
        {
            return null;
        }
        foreach (NConfirmButton button in buttons)
        {
            if (IsUnderPreviewContainer(button, screen))
            {
                return button;
            }
        }
        return buttons[0];
    }

    /// <summary>Is this button inside one of the screen's preview panels?</summary>
    private static bool IsUnderPreviewContainer(Node node, Node screen)
    {
        for (Node? walk = node; walk != null && walk != screen; walk = walk.GetParent())
        {
            if (walk.Name.ToString().Contains("Preview", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// The HAND is asking, and nothing is on the overlay stack.
    ///
    /// `CardSelectCmd.FromHand` does not open a screen -- it puts the hand
    /// itself into a selection mode (`NCombatRoom.Ui.Hand.SelectCards`,
    /// CardSelectCmd.cs:727) where you click cards in hand and press a
    /// checkmark. Everything this describer knew about lived on the overlay
    /// stack or was a room, so an in-hand selection was invisible to it:
    /// `Describe()` returned null, `ScheduleAdvice` fell through to the combat
    /// board, and every card it then tried to play was silently dropped by the
    /// action queue -- which is blocked precisely because a selection is
    /// pending. The result was a livelock, not a stall: 2,795 `Defend`s in
    /// 2h40m on floor 3 of act 2 (Ashwater potion, "choose any number of cards
    /// to exhaust", 2026-08-06), with the action counter ticking up the whole
    /// time so every watchdog read it as healthy.
    ///
    /// This is a large class of effects, not one potion: Burning Pact, Brand,
    /// Scavenge, True Grit, Tyranny, every discard-selection, and the in-hand
    /// upgrade mode all come through here.
    /// </summary>
    private static SimScreenDecision? DescribeHandSelect()
    {
        NPlayerHand? hand = NPlayerHand.Instance;
        if (hand == null || !GodotObject.IsInstanceValid(hand))
        {
            return null;
        }
        NPlayerHand.Mode mode = hand.CurrentMode;
        if (mode != NPlayerHand.Mode.SimpleSelect && mode != NPlayerHand.Mode.UpgradeSelect)
        {
            return null;
        }

        (int minSelect, int maxSelect) = ReadHandSelectionBounds(hand);
        int selected = HandSelectedCount(hand);
        NConfirmButton? confirm = HandConfirmButton(hand);

        // The confirm button IS the game's answer to "is this selection
        // legal yet" -- RefreshSelectModeConfirmButton enables it exactly when
        // MinSelect <= count <= MaxSelect (NPlayerHand.cs). So once it is live
        // and something is picked, committing is the decision.
        if (selected > 0 && confirm != null &&
            GodotObject.IsInstanceValid(confirm) && confirm.IsEnabled)
        {
            string label = selected == 1 ? "confirm" : $"confirm ({selected} cards)";
            return LocalDecision(
                NonCombatBridgeProtocol.CardSelectState, hand, "NPlayerHand",
                $"hand select: {label}", label,
                _ =>
                {
                    if (!GodotObject.IsInstanceValid(confirm) || !confirm.IsEnabled)
                    {
                        return false;
                    }
                    TaskHelper.RunSafely(UiHelper.Click(confirm));
                    return true;
                },
                signature: $"handselect:confirm:{selected}");
        }

        // A picked card LEAVES the hand (SelectCardInSimpleMode ->
        // RemoveCardHolder, into the selected-card container), so unlike the
        // grid screens there is nothing to filter out here: what is still in
        // hand is exactly what is still pickable. `ActiveHolders` is already
        // restricted to the visible ones, which is how the selection filter
        // expresses "this card does not qualify".
        List<NHandCardHolder> holders = hand.ActiveHolders
            .Where(h => GodotObject.IsInstanceValid(h) && h.CardNode?.Model != null)
            .ToList();
        if (holders.Count == 0)
        {
            return null;
        }

        var cards = new List<Dictionary<string, object?>>();
        var labels = new List<string>();
        for (int i = 0; i < holders.Count; ++i)
        {
            Dictionary<string, object?> data =
                SimStateExtractor.SerializeCard(holders[i].CardNode!.Model);
            data["index"] = i;
            cards.Add(data);
            labels.Add(holders[i].CardNode!.Model.Title);
        }

        (string? source, string? promptKey, string? sourceModel, bool? cancelable) =
            SimCardSelectContext.Peek();

        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(runState);
        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.CardSelectState,
            ["cards"] = cards,
            ["min_select"] = minSelect,
            ["max_select"] = maxSelect,
            ["min_count"] = minSelect,
            ["max_count"] = maxSelect,
            ["cancelable"] = cancelable ?? (minSelect <= 0),
            ["prompt_key"] = (object?)promptKey!,
            ["source"] = (object?)source!,
            ["source_model"] = (object?)sourceModel!,
            // The source model'"'"'s own detail: its KIND by runtime type (so a
            // client never has to infer "is this a card?" from whether the id
            // turns up in a pile), its target while the play is still in
            // flight, and whether the copy played was upgraded. See
            // SimCardSelectContext.SourceKind / SourceTargetIndex.
            ["source_kind"] = (object?)SimCardSelectContext.SourceKind!,
            ["source_target_index"] = (object?)SimCardSelectContext.SourceTargetIndex!,
            ["source_upgraded"] = (object?)SimCardSelectContext.SourceUpgraded!,
            ["screen"] = "NPlayerHand",
            ["combat"] = (object?)SimCombatHandler.BuildEmbeddedCombatState(me)!,
        };
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.CardSelectState,
            State = state,
            OptionCount = holders.Count,
            OptionLabels = labels,
            Screen = hand,
            ScreenName = "NPlayerHand",
            // `index == OptionCount` is the takeover's SKIP leg (SimAdvisor).
            Choose = index =>
                index >= holders.Count
                    ? ConfirmHandSelection(hand)
                    : ChooseHandCard(hand, index),
        };
    }

    /// <summary>
    /// Commit the hand selection AS IT STANDS -- the takeover's `skip` leg.
    ///
    /// <para>A hand select has no skip option to click: "select nothing" here
    /// means pressing CONFIRM with an empty selection. The takeover expresses a
    /// skip as `index == OptionCount` (SimAdvisor), a phantom option one past
    /// the end that every grid screen answers with its skip button -- and this
    /// screen could not answer at all: `could not pick option 1 on
    /// NPlayerHand`, then 180s of nothing until the stall guard killed the
    /// run.</para>
    ///
    /// <para>Unreachable until the advisor started SEARCHING end-of-turn
    /// retain screens. Well-Laid Plans offers `min_select` 0, so a search that
    /// decides to retain nothing answers with a skip -- which no in-combat
    /// selection had ever produced before (zero occurrences of `0 pick(s)
    /// searched` across the check10 and check11 batches, two in the first run
    /// after).</para>
    ///
    /// <para>`IsEnabled` is the game's own legality test --
    /// `RefreshSelectModeConfirmButton` enables the button exactly when
    /// MinSelect &lt;= count &lt;= MaxSelect (NPlayerHand.cs) -- so it needs no
    /// second opinion here. A skip the game considers illegal simply fails,
    /// and the takeover re-asks.</para>
    /// </summary>
    private static bool ConfirmHandSelection(NPlayerHand hand)
    {
        NConfirmButton? confirm = HandConfirmButton(hand);
        if (confirm == null || !GodotObject.IsInstanceValid(confirm) || !confirm.IsEnabled)
        {
            return false;
        }
        TaskHelper.RunSafely(UiHelper.Click(confirm));
        return true;
    }

    /// <summary>
    /// Pick a card in hand while the hand is in a selection mode.
    ///
    /// NOT `UiHelper.Click(holder.Hitbox)`: that emits the hitbox's `Released`
    /// signal, and a card holder does not listen to it. It listens to
    /// `MousePressed`/`MouseReleased` and to real input in `_GuiInput`, both of
    /// which end in `EmitPressed` -> the holder's own `Pressed` signal, which
    /// is what `NPlayerHand.OnHolderPressed` is connected to. Clicking the
    /// hitbox therefore SUCCEEDED and selected nothing -- three identical
    /// picks in a row until the runaway guard stopped autoplay. So emit the
    /// signal the holder actually publishes.
    /// </summary>
    private static bool ChooseHandCard(NPlayerHand hand, int index)
    {
        if (!GodotObject.IsInstanceValid(hand))
        {
            return false;
        }
        // Re-read the hand: the description was computed against a list that
        // may have moved (a card selected, a monster's power revalidating the
        // filter), and an index is only as good as the list it came from.
        List<NHandCardHolder> holders = hand.ActiveHolders
            .Where(h => GodotObject.IsInstanceValid(h) && h.CardNode?.Model != null)
            .ToList();
        if (index < 0 || index >= holders.Count)
        {
            return false;
        }
        NHandCardHolder holder = holders[index];
        int before = HandSelectedCount(hand);
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
        // Report what the game did, not what we asked for. A pick that does
        // not land is indistinguishable from one that does at this level, and
        // that is precisely how the hitbox version stayed invisible.
        int after = HandSelectedCount(hand);
        if (after == before)
        {
            Logger.Log(
                $"[SimScreenAdvice] Hand pick '{holder.CardNode!.Model.Title}' did not change " +
                $"the selection ({before} selected).");
            return false;
        }
        return true;
    }

    /// <summary>The hand's `%SelectModeConfirmButton` -- the checkmark.</summary>
    private static NConfirmButton? HandConfirmButton(NPlayerHand hand)
    {
        try
        {
            return hand.GetNodeOrNull<NConfirmButton>("%SelectModeConfirmButton")
                ?? UiHelper.FindFirst<NConfirmButton>(hand);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimScreenAdvice] Hand confirm unavailable: {ex.Message}");
            return null;
        }
    }

    /// <summary>How many cards the hand's selection already holds.</summary>
    private static int HandSelectedCount(NPlayerHand hand)
    {
        try
        {
            FieldInfo? field = typeof(NPlayerHand).GetField(
                "_selectedCards", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field?.GetValue(hand) is System.Collections.IEnumerable selected)
            {
                int count = 0;
                foreach (object? _ in selected)
                {
                    count++;
                }
                return count;
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimScreenAdvice] Hand selection count unavailable: {ex.Message}");
        }
        return 0;
    }

    /// <summary>
    /// The hand's active `CardSelectorPrefs` bounds. Falls back to (1, 1) --
    /// the shape of nearly every in-hand selection -- but note the fallback is
    /// only ever a description detail here: what may be committed is decided by
    /// the game's own confirm button, not by these numbers.
    /// </summary>
    private static (int, int) ReadHandSelectionBounds(NPlayerHand hand)
    {
        try
        {
            FieldInfo? field = typeof(NPlayerHand).GetField(
                "_prefs", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field?.GetValue(hand) is CardSelectorPrefs prefs)
            {
                return (prefs.MinSelect, prefs.MaxSelect);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimScreenAdvice] Hand selection bounds unavailable: {ex.Message}");
        }
        return (1, 1);
    }

    private static SimScreenDecision? DescribeCardSelect(NCardGridSelectionScreen screen)
    {
        // Once something is selected, the question is no longer "which card" --
        // it is "commit this". Describing the grid again here is what let
        // autoplay pick a second, third, ... card on a screen that was only
        // ever waiting to be confirmed (59 straight `Takeover executed on
        // NDeckEnchantSelectScreen: Defend` in godot.log, 2026-08-06).
        int selected = SelectedCount(screen);
        (int minSelect, int maxSelect) = ReadSelectionBounds(screen);
        if (selected > 0)
        {
            NConfirmButton? confirm = PendingConfirm(screen);
            if (confirm != null)
            {
                string label = selected == 1 ? "confirm" : $"confirm ({selected} cards)";
                return LocalDecision(
                    NonCombatBridgeProtocol.CardSelectState, screen, screen.GetType().Name,
                    $"card select: {label}", label,
                    _ =>
                    {
                        if (!GodotObject.IsInstanceValid(confirm) || !confirm.IsEnabled)
                        {
                            return false;
                        }
                        TaskHelper.RunSafely(UiHelper.Click(confirm));
                        return true;
                    });
            }
            if (selected >= maxSelect)
            {
                // Picked everything it asked for, but nothing to press yet:
                // the preview is still animating in. A wait, not a decision.
                return null;
            }
            // Otherwise this screen is not waiting to be confirmed, it is
            // waiting for the REST of the selection. A screen with
            // MinSelect == MaxSelect never enables a confirm at ALL
            // (RefreshConfirmButtonVisibility gates on MinSelect != MaxSelect,
            // NDeckCardSelectScreen.cs:241) and opens its own preview once the
            // last card is picked -- so on the Amalgamator's "combine two
            // Strikes" the branch above found no button, this describer
            // returned null, and the screen dropped out of the decision stream
            // entirely: a permanent stall on floor 21, 2026-08-06. Fall
            // through and offer what is still pickable.
        }

        // Already-selected cards are excluded: `_selectedCards` is a HashSet,
        // so re-picking one DE-selects it, and an autoplay loop that keeps
        // being offered its own last pick can toggle a card on and off forever.
        List<NGridCardHolder> holders = SelectableHolders(screen);
        if (holders.Count == 0)
        {
            return null;
        }

        var cards = new List<Dictionary<string, object?>>();
        var labels = new List<string>();
        for (int i = 0; i < holders.Count; ++i)
        {
            Dictionary<string, object?> data = SimStateExtractor.SerializeCard(holders[i].CardModel!);
            data["index"] = i;
            cards.Add(data);
            labels.Add(holders[i].CardModel!.Title);
        }

        // The prompt key is the authoritative "what does picking this MEAN"
        // signal (TO_UPGRADE / TO_REMOVE / TO_TRANSFORM ...) and the Python
        // side keys its policy surface off it. CardSelectContextPatch records
        // it for every CardSelectCmd entry point, including the ones that
        // never reach ICardSelector -- which is all of them in advisor mode --
        // so PEEK it here rather than consuming it.
        (string? source, string? promptKey, string? sourceModel, bool? cancelable) =
            SimCardSelectContext.Peek();

        RunState? runState = RunManager.Instance.DebugOnlyGetState();
        Player? me = LocalContext.GetMe(runState);
        Dictionary<string, object> state = new()
        {
            [NonCombatBridgeProtocol.TypeField] = NonCombatBridgeProtocol.CardSelectState,
            ["cards"] = cards,
            ["min_select"] = minSelect,
            ["max_select"] = maxSelect,
            ["min_count"] = minSelect,
            ["max_count"] = maxSelect,
            ["cancelable"] = cancelable ?? (minSelect <= 0),
            ["prompt_key"] = (object?)promptKey!,
            ["source"] = (object?)source!,
            ["source_model"] = (object?)sourceModel!,
            // The source model'"'"'s own detail: its KIND by runtime type (so a
            // client never has to infer "is this a card?" from whether the id
            // turns up in a pile), its target while the play is still in
            // flight, and whether the copy played was upgraded. See
            // SimCardSelectContext.SourceKind / SourceTargetIndex.
            ["source_kind"] = (object?)SimCardSelectContext.SourceKind!,
            ["source_target_index"] = (object?)SimCardSelectContext.SourceTargetIndex!,
            ["source_upgraded"] = (object?)SimCardSelectContext.SourceUpgraded!,
            ["screen"] = screen.GetType().Name,
            ["combat"] = (object?)SimCombatHandler.BuildEmbeddedCombatState(me)!,
        };
        SimStateExtractor.ApplyEnvelope(state, runState, me);

        return new SimScreenDecision
        {
            AdviceFor = NonCombatBridgeProtocol.CardSelectState,
            State = state,
            OptionCount = holders.Count,
            OptionLabels = labels,
            Screen = screen,
            ScreenName = screen.GetType().Name,
            OptionRefs = holders.Select(h => (object?)h.CardModel).ToList(),
            // `index == OptionCount` is the takeover's SKIP leg (SimAdvisor).
            Choose = index =>
                index >= holders.Count
                    ? ConfirmGridSelection(screen)
                    : ChooseGridCard(screen, index),
        };
    }

    /// <summary>
    /// Commit the grid selection AS IT STANDS -- the takeover's `skip` leg.
    ///
    /// <para>"Choose UP TO 2 cards" is `min_select` 0, so a search may
    /// legitimately answer "take none", and the takeover expresses that as
    /// `index == OptionCount` (SimAdvisor) -- a phantom option one past the
    /// end. `ChooseGridCard` returns false for it, so the screen could not be
    /// answered at all: `could not pick option 1 on
    /// NCombatPileCardSelectScreen` with one card offered, then 180s until
    /// the stall guard killed run 005-silent-a8-0825-1344.</para>
    ///
    /// <para>The hand select grew the same leg one commit earlier
    /// (4a6838907) and this is the identical defect on the OTHER screen
    /// family -- unreachable until the advisor started searching
    /// `min_select` 0 selections, so fixing one screen only moved the stall.
    /// </para>
    ///
    /// <para><see cref="PendingConfirm"/> already carries the part that is
    /// actually hard here: these screens do not share one confirm button, and
    /// looking only for `%Confirm` finds a disabled button or none at all on
    /// three of the four topologies (see its remarks). It also filters to
    /// enabled+visible, which is the game's own legality test, so a skip the
    /// game considers illegal simply fails and the takeover re-asks.</para>
    /// </summary>
    private static bool ConfirmGridSelection(NCardGridSelectionScreen screen)
    {
        if (!(GodotObject.IsInstanceValid(screen) && screen.IsVisibleInTree()))
        {
            return false;
        }
        NConfirmButton? confirm = PendingConfirm(screen);
        if (confirm == null)
        {
            return false;
        }
        TaskHelper.RunSafely(UiHelper.Click(confirm));
        return true;
    }

    private static bool ChooseGridCard(NCardGridSelectionScreen screen, int index)
    {
        if (!(GodotObject.IsInstanceValid(screen) && screen.IsVisibleInTree()))
        {
            return false;
        }
        NCardGrid? grid = UiHelper.FindFirst<NCardGrid>(screen);
        if (grid == null || !GodotObject.IsInstanceValid(grid))
        {
            return false;
        }
        // Same filter the description used -- an index is only as good as the
        // list it was computed against.
        List<NGridCardHolder> holders = SelectableHolders(screen);
        if (index < 0 || index >= holders.Count)
        {
            return false;
        }

        grid.EmitSignal(NCardGrid.SignalName.HolderPressed, holders[index]);

        // Screens that require manual confirmation stay open until a confirm
        // button is pressed; the ones that don't close themselves on the pick.
        // Confirming is normally the NEXT decision -- DescribeCardSelect sees
        // the selection and offers it -- which keeps exactly one code path
        // pressing the button. That relies on being able to read the screen's
        // selection, so screens whose `_selectedCards` we cannot see (a future
        // shape) fall back to settling the pick here instead of stalling.
        if (SelectedCount(screen) < 0)
        {
            TaskHelper.RunSafely(ConfirmUnknownScreenAsync(screen));
        }
        return true;
    }

    /// <summary>
    /// Last-resort confirm for a card grid whose selection state we cannot
    /// read. Presses whatever confirm the screen offers, waiting for the pick
    /// (and any preview it opens) to settle. Bounded, because an unbounded
    /// retry on a screen that will not close is how this bug started.
    /// </summary>
    private static async System.Threading.Tasks.Task ConfirmUnknownScreenAsync(
        NCardGridSelectionScreen screen)
    {
        for (int attempt = 0; attempt < 6; attempt++)
        {
            // The game's own handlers use the same 200-300 ms beat before
            // reading IsEnabled.
            await System.Threading.Tasks.Task.Delay(300);
            if (!(GodotObject.IsInstanceValid(screen) && screen.IsVisibleInTree()))
            {
                return;
            }
            NConfirmButton? confirm = PendingConfirm(screen);
            if (confirm != null)
            {
                await UiHelper.Click(confirm);
            }
        }
    }

    /// <summary>
    /// (min, max) selection counts. `CardSelectorPrefs` lives in a private
    /// field on the concrete screen (`_prefs`), so this is reflection with a
    /// single-pick default -- a wrong guess only mislabels the request, it
    /// cannot mis-click.
    /// </summary>
    private static (int, int) ReadSelectionBounds(NCardGridSelectionScreen screen)
    {
        try
        {
            FieldInfo? field = screen.GetType().GetField(
                "_prefs", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field?.GetValue(screen) is CardSelectorPrefs prefs)
            {
                return (prefs.MinSelect, prefs.MaxSelect);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimScreenAdvice] Selection bounds unavailable: {ex.Message}");
        }
        return (1, 1);
    }
}
