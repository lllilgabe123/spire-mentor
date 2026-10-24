// Bot.cs -- SpireTestBot: plays random legal moves for the LOCAL player so two
// game instances on one PC can run a multiplayer game unattended.
//
// Inert unless launched with -spiretestbot. Options:
//   -spiretestbot-out <file>     JSON-lines status log (what the test script reads)
//   -spiretestbot-floors <n>     stop once the run reaches floor n (default 6)
//   -spiretestbot-probe          host only: once, in combat, trigger SpireMentor's
//                                AI / undo / speed actions to check they are refused
//   -spiretestbot-cooptrainer    host only: once, on the map, use the Host Tools
//                                ("Spire Co-op Trainer") mod to give the host
//                                itself gold, a card and a relic
//   -spiretestbot-cooptrainer-card <ID>   that card instead of a random one
//                                (e.g. MIDNIGHT, an MP-only beta card)
//   -spiretestbot-cooptrainer-other       give to the OTHER player instead
//                                (Host Tools' "experimental" remote rewards)
//   -spiretestbot-follow-ai      host only: in fights, play SpireMentor's current
//                                recommendation (its search) instead of the
//                                simple rules, so test runs reach elites and
//                                bosses and the tracker gets "followed the AI"
//                                co-op fights. TEST HARNESS ONLY: SpireMentor
//                                itself never acts for the player in co-op.
// Both roles log a "snapshot" of every player's gold / deck / relics at each
// map, so the script can check the two games agree.
//
// Everything it does goes through the same paths a human uses, so the other
// instance sees ordinary multiplayer traffic:
//   - UI clicks (UiHelper.Click / button signals) for menus, map, events, rooms;
//   - ActionQueueSynchronizer.RequestEnqueue(PlayCardAction) for card plays;
//   - RequestEnqueue(EndPlayerTurnAction) for ending the turn, as NEndTurnButton.
// It deliberately does NOT use the game's AutoSlay combat/event handlers (they
// grant powers and kill enemies directly, which only changes THIS instance)
// nor CardSelectCmd.UseSelector (it bypasses the network-synchronised card
// selection). The AutoSlay SCREEN handlers are reused: they only click.

using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Handlers.Screens;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.addons.mega_text;

namespace SpireTestBot;

[ModInitializer(nameof(Initialize))]
public static class Bot
{
    private const string RunRoot = "/root/Game/RootSceneContainer/Run/RoomContainer/";

    private static readonly object _fileLock = new();
    private static readonly System.Random _r = new();
    private static readonly Rng _rng = new((ulong)System.Environment.TickCount64);

    private static string? _outPath;
    private static int _targetFloor = 6;
    private static bool _probe;
    private static bool _coopTrainer;
    private static bool _coopTrainerDone;
    private static string? _coopTrainerCard;
    private static bool _coopTrainerOther;
    private static bool _followAi;
    private static DateTime _aiWaitSince = DateTime.MinValue;
    private static int _aiMoves;
    private static int _ruleMoves;
    private static int _snapshotFloor = -1;
    private static bool _isHost;
    private static string Role => _isHost ? "host" : "client";

    private static readonly Dictionary<Type, IScreenHandler> _screenHandlers = new()
    {
        [typeof(NRewardsScreen)] = new RewardsScreenHandler(),
        [typeof(NCardRewardSelectionScreen)] = new CardRewardScreenHandler(),
        [typeof(NDeckUpgradeSelectScreen)] = new DeckUpgradeScreenHandler(),
        [typeof(NDeckTransformSelectScreen)] = new DeckTransformScreenHandler(),
        [typeof(NDeckEnchantSelectScreen)] = new DeckEnchantScreenHandler(),
        [typeof(NDeckCardSelectScreen)] = new DeckCardSelectScreenHandler(),
        [typeof(NSimpleCardSelectScreen)] = new SimpleCardSelectScreenHandler(),
        [typeof(NChooseACardSelectionScreen)] = new ChooseACardScreenHandler(),
        [typeof(NChooseABundleSelectionScreen)] = new ChooseABundleScreenHandler(),
        [typeof(NChooseARelicSelection)] = new ChooseARelicScreenHandler(),
        [typeof(NCrystalSphereScreen)] = new CrystalSphereScreenHandler(),
    };

    // Progress / stuck tracking.
    private static DateTime _lastProgress = DateTime.UtcNow;
    private static DateTime _lastStuckReport = DateTime.MinValue;
    private static string _lastNote = "";

    // Per-phase state.
    private static bool _embarked;
    private static int _lastLobbyCount = -1;
    private static bool _runSeen;
    private static bool _done;
    private static bool _probed;
    private static int _turn = -1;
    private static int _potionTurn = -1;
    private static int _cardsThisTurn;
    private static readonly HashSet<CardModel> _attempted = new();
    private static DateTime _lastMapClick = DateTime.MinValue;
    private static DateTime _lastEventClick = DateTime.MinValue;
    private static DateTime _lastRoomClick = DateTime.MinValue;
    private static int _restChosenFloor = -1;
    private static int _chestOpenedFloor = -1;
    private static int _relicPickedFloor = -1;
    private static readonly HashSet<string> _seenModals = new();
    private static readonly HashSet<string> _seenScreens = new();

    public static void Initialize()
    {
        if (!CommandLineHelper.HasArg("spiretestbot"))
        {
            return;
        }
        _outPath = CommandLineHelper.GetValue("spiretestbot-out");
        if (int.TryParse(CommandLineHelper.GetValue("spiretestbot-floors"), out int floors))
        {
            _targetFloor = floors;
        }
        _probe = CommandLineHelper.HasArg("spiretestbot-probe");
        _coopTrainer = CommandLineHelper.HasArg("spiretestbot-cooptrainer");
        _coopTrainerCard = CommandLineHelper.GetValue("spiretestbot-cooptrainer-card");
        _coopTrainerOther = CommandLineHelper.HasArg("spiretestbot-cooptrainer-other");
        _followAi = CommandLineHelper.HasArg("spiretestbot-follow-ai");
        _isHost = CommandLineHelper.GetValue("fastmp")?.StartsWith("host") ?? false;
        Status("started", new { target_floor = _targetFloor, probe = _probe });
        TaskHelper.RunSafely(StartWhenReady());
    }

    private static async Task StartWhenReady()
    {
        while (NGame.Instance == null)
        {
            await Task.Delay(250);
        }
        Callable.From(() => TaskHelper.RunSafely(Loop())).CallDeferred();
    }

    private static async Task Loop()
    {
        try
        {
            // Fresh test profiles would otherwise stop on tutorial popups.
            MegaCrit.Sts2.Core.Saves.SaveManager.Instance.SetFtuesEnabled(false);
        }
        catch (Exception ex)
        {
            Status("error", new { state = "ftue", message = ex.Message });
        }
        while (!_done)
        {
            bool acted = false;
            try
            {
                acted = await Tick();
                if (acted)
                {
                    _lastProgress = DateTime.UtcNow;
                }
                ReportIfStuck();
            }
            catch (Exception ex)
            {
                Status("error", new { state = Describe(), message = ex.GetType().Name + ": " + ex.Message, stack = ex.StackTrace });
                await Task.Delay(1000);
            }
            await Task.Delay(acted ? 250 : 500);
        }
    }

    // One look at the screen, at most one action. Returns true if it acted.
    private static async Task<bool> Tick()
    {
        Node root = ((SceneTree)Engine.GetMainLoop()).Root;

        if (await HandleModal())
        {
            return true;
        }

        RunManager rm = RunManager.Instance;
        if (rm == null || !rm.IsInProgress)
        {
            if (_runSeen)
            {
                Status("left_run", new { state = Describe() });
                Finish();
                return false;
            }
            return await HandleLobby(root);
        }

        RunState? state = rm.DebugOnlyGetState();
        if (state == null)
        {
            return false;
        }
        if (!_runSeen)
        {
            _runSeen = true;
            Status("run_started", new { net = rm.NetService?.Type.ToString(), players = state.Players.Count });
        }

        IOverlayScreen? top = NOverlayStack.Instance?.Peek();
        // After Proceed the map opens while the finished rewards screen is
        // still on the stack (AutoSlayer's drain loop special-cases this too).
        if (top is NRewardsScreen && (NMapScreen.Instance?.IsOpen ?? false))
        {
            top = null;
        }
        if (top is Node topNode)
        {
            if (topNode is NGameOverScreen)
            {
                Status("game_over", new { floor = state.TotalFloor });
                Finish();
                return false;
            }
            if (_screenHandlers.TryGetValue(topNode.GetType(), out IScreenHandler? handler))
            {
                Note("screen " + topNode.GetType().Name);
                using var cts = new CancellationTokenSource(handler.Timeout);
                await handler.HandleAsync(_rng, cts.Token);
                return true;
            }
            if (_seenScreens.Add(topNode.GetType().Name))
            {
                Status("unknown_screen", new { type = topNode.GetType().Name });
            }
            if (await HandleGenericScreen(topNode))
            {
                return true;
            }
        }

        if (await HandleHandSelection())
        {
            return true;
        }
        if (CombatManager.Instance.IsInProgress)
        {
            return await HandleCombat(state);
        }

        NMapScreen? map = NMapScreen.Instance;
        if (map != null && map.IsOpen)
        {
            if (_snapshotFloor != state.TotalFloor)
            {
                _snapshotFloor = state.TotalFloor;
                Snapshot(state);
            }
            if (_coopTrainer && _isHost && !_coopTrainerDone && state.TotalFloor >= 1)
            {
                await CoopTrainerTest(state);
                Snapshot(state);
                return true;
            }
            if (state.TotalFloor >= _targetFloor)
            {
                Status("done", new { floor = state.TotalFloor });
                Finish();
                return false;
            }
            return await HandleMap(map, state);
        }

        return await HandleRoom(root, state);
    }

    // ---------------------------------------------------------------- lobby

    private static async Task<bool> HandleLobby(Node root)
    {
        NCharacterSelectScreen? screen = UiHelper.FindFirst<NCharacterSelectScreen>(root);
        if (screen == null || !screen.IsVisibleInTree())
        {
            return false;
        }
        int players = screen.Lobby?.Players.Count ?? 0;
        if (players != _lastLobbyCount)
        {
            _lastLobbyCount = players;
            Status("lobby", new { players });
        }
        if (_embarked || (_isHost && players < 2))
        {
            return false;
        }
        List<NCharacterSelectButton> buttons = UiHelper.FindAll<NCharacterSelectButton>(screen)
            .Where(b => !b.IsLocked && b.IsVisibleInTree()).ToList();
        if (buttons.Count == 0)
        {
            return false;
        }
        NCharacterSelectButton pick = buttons[_r.Next(buttons.Count)];
        pick.Select();
        await Task.Delay(400);
        NConfirmButton? confirm = screen.GetNodeOrNull<NConfirmButton>("ConfirmButton")
            ?? UiHelper.FindAll<NConfirmButton>(screen).FirstOrDefault(b => b.IsVisibleInTree());
        if (confirm == null || !confirm.IsEnabled)
        {
            return false;
        }
        await UiHelper.Click(confirm);
        _embarked = true;
        Status("embarked", new { character = pick.Character?.Id.Entry, players });
        return true;
    }

    // --------------------------------------------------------------- modals

    private static async Task<bool> HandleModal()
    {
        if (NModalContainer.Instance?.OpenModal is not Node modal || !GodotObject.IsInstanceValid(modal))
        {
            return false;
        }
        string type = modal.GetType().Name;
        string text = CollectText(modal);
        if (_seenModals.Add(type + "|" + text))
        {
            Status("modal", new { type, text });
        }
        if (modal is NErrorPopup)
        {
            // Connection errors (mod mismatch, disconnect) end the test; the
            // script reads the text from the status file.
            if (_runSeen || _embarked || DateTime.UtcNow - _lastProgress > TimeSpan.FromSeconds(20))
            {
                Status("error_popup", new { text });
                Finish();
            }
            return false;
        }
        // First-launch notices (EA disclaimer, modded-save notice, ...): press
        // the affirmative button.
        NClickableControl? ok = UiHelper.FindAll<NClickableControl>(modal)
            .Where(b => b.IsVisibleInTree() && b.IsEnabled)
            .OrderByDescending(b => Score(b.Name.ToString()))
            .FirstOrDefault(b => Score(b.Name.ToString()) > 0);
        if (ok == null)
        {
            return false;
        }
        Note($"modal {type}: clicking {ok.Name}");
        await UiHelper.Click(ok);
        return true;

        static int Score(string name)
        {
            string n = name.ToLowerInvariant();
            if (n.Contains("no") && !n.Contains("notice")) return 0;
            if (n.Contains("abandon") || n.Contains("cancel") || n.Contains("back")) return 0;
            if (n.Contains("yes") || n.Contains("confirm") || n.Contains("ok") || n.Contains("accept")) return 3;
            if (n.Contains("continue") || n.Contains("dismiss") || n.Contains("close")) return 2;
            return 0;
        }
    }

    // Screens AutoSlay has no handler for (e.g. NCombatPileCardSelectScreen,
    // Headbutt's "put a discarded card on top"): any card grid is driven like
    // SimpleCardSelectScreenHandler (press a card, then Confirm); otherwise a
    // Confirm / Proceed / Back button is the way out.
    private static async Task<bool> HandleGenericScreen(Node screen)
    {
        NConfirmButton? confirm = UiHelper.FindAll<NConfirmButton>(screen)
            .FirstOrDefault(b => b.IsVisibleInTree() && b.IsEnabled);
        if (confirm != null)
        {
            Note("generic screen: confirm on " + screen.GetType().Name);
            await UiHelper.Click(confirm);
            return true;
        }
        if (screen is NCardGridSelectionScreen && UiHelper.FindFirst<NCardGrid>(screen) is NCardGrid grid)
        {
            List<NGridCardHolder> holders = grid.CurrentlyDisplayedCardHolders.Where(h => h.IsVisibleInTree()).ToList();
            if (holders.Count > 0 && DateTime.UtcNow - _lastRoomClick > TimeSpan.FromSeconds(1))
            {
                _lastRoomClick = DateTime.UtcNow;
                Note("generic screen: pick a card on " + screen.GetType().Name);
                grid.EmitSignal(NCardGrid.SignalName.HolderPressed, holders[_r.Next(holders.Count)]);
                await Task.Delay(300);
                return true;
            }
        }
        NClickableControl? way = UiHelper.FindAll<NProceedButton>(screen).Cast<NClickableControl>()
            .Concat(UiHelper.FindAll<NBackButton>(screen))
            .FirstOrDefault(b => b.IsVisibleInTree() && b.IsEnabled);
        if (way != null && DateTime.UtcNow - _lastProgress > TimeSpan.FromSeconds(5))
        {
            Note("generic screen: " + way.Name + " on " + screen.GetType().Name);
            await UiHelper.Click(way);
            return true;
        }
        return false;
    }

    // --------------------------------------------------------------- combat

    private static async Task<bool> HandleHandSelection()
    {
        NPlayerHand? hand = NPlayerHand.Instance;
        if (hand == null || !hand.IsInCardSelection)
        {
            return false;
        }
        NConfirmButton? confirm = UiHelper.FindAll<NConfirmButton>(hand)
            .FirstOrDefault(b => b.IsVisibleInTree() && b.IsEnabled);
        if (confirm != null)
        {
            Note("hand selection: confirm");
            await UiHelper.Click(confirm);
            return true;
        }
        List<NHandCardHolder> holders = hand.ActiveHolders.ToList();
        if (holders.Count == 0)
        {
            return false;
        }
        NHandCardHolder holder = holders[_r.Next(holders.Count)];
        Note("hand selection: pick a card");
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
        await Task.Delay(300);
        return true;
    }

    private static async Task<bool> HandleCombat(RunState state)
    {
        CombatManager cm = CombatManager.Instance;
        Player? me = LocalContext.GetMe(state);
        PlayerCombatState? pcs = me?.PlayerCombatState;
        if (me == null || pcs == null || pcs.Phase != PlayerTurnPhase.Play)
        {
            return false;
        }
        if (cm.IsEnding || cm.PlayerActionsDisabled || cm.IsPlayerReadyToEndTurn(me))
        {
            return false;
        }
        if (RunManager.Instance.ActionExecutor.CurrentlyRunningAction != null)
        {
            return false;
        }
        if (pcs.TurnNumber != _turn)
        {
            _turn = pcs.TurnNumber;
            _attempted.Clear();
            _cardsThisTurn = 0;
        }

        if (_probe && _isHost && !_probed && state.TotalFloor >= 1)
        {
            await Probe();
            return true;
        }

        if (_followAi && _isHost)
        {
            bool? ai = await TryAiMove(me);
            if (ai != null)
            {
                return ai.Value; // played the AI's move, or still waiting for it
            }
        }
        _ruleMoves++;

        if (await TryUsePotion(me))
        {
            return true;
        }

        List<CardModel> playable = PileType.Hand.GetPile(me).Cards
            .Where(c => !_attempted.Contains(c) && c.CanPlay(out UnplayableReason _, out AbstractModel? _))
            .ToList();
        // Not clever, just not random: powers first (they pay off all fight),
        // then attacks (a dead enemy deals no damage), then the rest.
        foreach (CardType first in new[] { CardType.Power, CardType.Attack })
        {
            List<CardModel> preferred = playable.Where(c => c.Type == first).ToList();
            if (preferred.Count > 0)
            {
                playable = preferred;
                break;
            }
        }
        if (playable.Count == 0 || _cardsThisTurn >= 25)
        {
            Note($"turn {_turn}: end turn after {_cardsThisTurn} cards");
            // What NEndTurnButton does. PlayerCmd.EndTurn only marks the turn
            // ended on THIS instance; the other player never hears of it.
            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(me, pcs.TurnNumber));
            await Task.Delay(500);
            return true;
        }

        CardModel card = playable[_r.Next(playable.Count)];
        _attempted.Add(card);
        Creature? target = PickTarget(card, me);
        if (card.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer && target == null)
        {
            return true; // no legal target; try another card next tick
        }
        Note($"turn {_turn}: play {card.Id.Entry}");
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(card, target));
        _cardsThisTurn++;
        await Task.Delay(400);
        return true;
    }

    // SpireMentor's standing recommendation (SimAdvisor._lastBest, a
    // SimBestAction): non-null exactly while fresh advice stands for the
    // board on screen. Played through the same synced actions as the rules
    // below. Returns null to fall back to the rules (no advice within 6 s,
    // or one that cannot be played), true when it acted or is waiting.
    private static async Task<bool?> TryAiMove(Player me)
    {
        const System.Reflection.BindingFlags Any =
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static |
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;
        Type? advisorT = AppDomain.CurrentDomain.GetAssemblies()
            .Select(a => a.GetType("SpireMentor.SimAdvisor")).FirstOrDefault(t => t != null);
        // `public static readonly SimAdvisor Instance` -- a field, not a property.
        object? advisor = advisorT?.GetField("Instance", Any)?.GetValue(null)
                          ?? advisorT?.GetProperty("Instance", Any)?.GetValue(null);
        System.Reflection.FieldInfo? field = advisorT?.GetField("_lastBest", Any);
        object? best = advisor == null ? null : field?.GetValue(advisor);
        if (best == null)
        {
            if (_aiWaitSince == DateTime.MinValue)
            {
                _aiWaitSince = DateTime.UtcNow;
            }
            if (DateTime.UtcNow - _aiWaitSince < TimeSpan.FromSeconds(6))
            {
                return true; // the search usually answers within a second
            }
            _aiWaitSince = DateTime.MinValue;
            return null;
        }
        _aiWaitSince = DateTime.MinValue;
        field!.SetValue(advisor, null); // consume: never play the same advice twice
        Type bt = best.GetType();
        string action = (string)bt.GetField("Action")!.GetValue(best)!;
        int cardIndex = (int)bt.GetField("CardIndex")!.GetValue(best)!;
        int targetIndex = (int)bt.GetField("TargetIndex")!.GetValue(best)!;
        int slot = (int)bt.GetField("Slot")!.GetValue(best)!;
        Type? handler = advisorT!.Assembly.GetType("SpireMentor.SimCombatHandler");
        switch (action)
        {
            case "play":
            {
                IReadOnlyList<CardModel> hand = PileType.Hand.GetPile(me).Cards;
                if (cardIndex < 0 || cardIndex >= hand.Count)
                {
                    return null;
                }
                CardModel card = hand[cardIndex];
                if (!card.CanPlay(out UnplayableReason _, out AbstractModel? _))
                {
                    return null;
                }
                var target = (Creature?)handler?.GetMethod("ResolveTarget", Any)?.Invoke(null, new object[] { card, targetIndex });
                if (card.TargetType is TargetType.AnyEnemy && target == null)
                {
                    return null;
                }
                Note($"turn {_turn}: AI plays {card.Id.Entry}");
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(card, target));
                break;
            }
            case "end_turn":
                Note($"turn {_turn}: AI ends the turn");
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                    new EndPlayerTurnAction(me, me.PlayerCombatState!.TurnNumber));
                break;
            case "potion":
            {
                var potion = (PotionModel?)handler?.GetMethod("SafeGetPotionAtSlot", Any)?.Invoke(null, new object[] { me, slot });
                if (potion == null)
                {
                    return null;
                }
                var target = (Creature?)handler!.GetMethod("ResolvePotionTarget", Any)!.Invoke(null, new object?[] { me, potion, targetIndex });
                Note($"turn {_turn}: AI drinks {potion.Id.Entry}");
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new UsePotionAction(potion, target, true));
                break;
            }
            default:
                return null;
        }
        _aiMoves++;
        _cardsThisTurn++;
        await Task.Delay(500);
        return true;
    }

    private static Creature? PickTarget(CardModel card, Player me)
    {
        var cs = card.CombatState;
        if (cs == null)
        {
            return null;
        }
        return PickTarget(card.TargetType, cs, me);
    }

    private static Creature? PickTarget(TargetType type, ICombatState cs, Player me)
    {
        if (type == TargetType.AnyEnemy)
        {
            // Focus the weakest enemy: fewer attackers sooner.
            return cs.HittableEnemies.Where(c => c.IsAlive).OrderBy(c => c.CurrentHp).FirstOrDefault();
        }
        List<Creature> pool = type switch
        {
            TargetType.AnyAlly => cs.PlayerCreatures.Where(c => c.IsAlive && c != me.Creature).ToList(),
            TargetType.AnyPlayer or TargetType.Self => new List<Creature> { me.Creature },
            _ => new List<Creature>(),
        };
        return pool.Count == 0 ? null : pool[_r.Next(pool.Count)];
    }

    // Below 40% HP, drink one usable potion per turn (UsePotionAction is what
    // the potion button enqueues).
    private static async Task<bool> TryUsePotion(Player me)
    {
        if (me.Creature.CurrentHp * 10 >= me.Creature.MaxHp * 4 || _potionTurn == _turn)
        {
            return false;
        }
        ICombatState? cs = me.Creature.CombatState;
        PotionModel? potion = me.Potions.FirstOrDefault(p =>
            p.Usage is PotionUsage.CombatOnly or PotionUsage.AnyTime && p.PassesCustomUsabilityCheck && !p.IsQueued);
        if (potion == null || cs == null)
        {
            return false;
        }
        Creature? target = PickTarget(potion.TargetType, cs, me);
        if (potion.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer && target == null)
        {
            return false;
        }
        _potionTurn = _turn;
        Note($"turn {_turn}: drink {potion.Id.Entry} at {me.Creature.CurrentHp}/{me.Creature.MaxHp} HP");
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new UsePotionAction(potion, target, true));
        await Task.Delay(500);
        return true;
    }

    // SpireMentor's AI / rewind / speed verbs must all be refused in a
    // multiplayer run; the test script checks godot.log for the refusals.
    // Called through SpireMentor.SimActions -- what its hotkeys and toolbar
    // call -- because synthetic key events are dropped by NHotkeyManager
    // unless the window has focus, and two test windows cannot both have it.
    private static async Task Probe()
    {
        _probed = true;
        Status("probe_start", new { });
        Type? actions = AppDomain.CurrentDomain.GetAssemblies()
            .Select(a => a.GetType("SpireMentor.SimActions"))
            .FirstOrDefault(t => t != null);
        if (actions == null)
        {
            Status("probe_error", new { message = "SpireMentor.SimActions not found (is Spire Mentor loaded?)" });
            return;
        }
        foreach ((string method, object[] args, string what) in new[]
                 {
                     ("AiOnce", new object[] { "probe: AI once" }, "ai_once"),
                     ("ToggleAutoplay", new object[] { "probe: AI auto" }, "ai_auto"),
                     ("UndoTurn", Array.Empty<object>(), "undo_turn"),
                     ("RestartRoom", Array.Empty<object>(), "restart_room"),
                     ("NudgeSpeed", new object[] { 0.5 }, "speed_up"),
                 })
        {
            try
            {
                actions.GetMethod(method, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!
                    .Invoke(null, args);
                Status("probe_call", new { what });
            }
            catch (Exception ex)
            {
                Status("probe_error", new { what, message = (ex.InnerException ?? ex).Message });
            }
            await Task.Delay(1500);
        }
        Status("probe_done", new { engine_time_scale = Engine.TimeScale });
    }

    // ------------------------------------------------------- co-op trainer

    private static void Snapshot(RunState state)
    {
        Status("snapshot", new
        {
            floor = state.TotalFloor,
            players = state.Players.Select(p => new
            {
                id = p.NetId,
                gold = p.Gold,
                deck = p.Deck.Cards.Count,
                relics = p.Relics.Count,
                relic_ids = string.Join(",", p.Relics.Select(r => r.Id.Entry)),
            }).ToList(),
        });
    }

    // Drives the Host Tools mod (HostGold.Entry) the way its panel does:
    // recipients = the host only, then its own AddGold / AddSelected tasks.
    private static async Task CoopTrainerTest(RunState state)
    {
        _coopTrainerDone = true;
        const System.Reflection.BindingFlags S =
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;
        const System.Reflection.BindingFlags I =
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public;
        Type? entry = AppDomain.CurrentDomain.GetAssemblies()
            .Select(a => a.GetType("HostGold.Entry")).FirstOrDefault(t => t != null);
        if (entry == null)
        {
            Status("coop_trainer_error", new { message = "HostGold.Entry not found (Host Tools not loaded?)" });
            return;
        }
        Player host = LocalContext.GetMe(state)!;
        // The recipient: the host itself, or (with -other) the other player.
        Player me = _coopTrainerOther
            ? state.Players.FirstOrDefault(p => p.NetId != host.NetId) ?? host
            : host;
        var before = new { recipient = me.NetId, gold = me.Gold, deck = me.Deck.Cards.Count, relics = me.Relics.Count };
        var results = new List<object>();
        try
        {
            var selected = (HashSet<ulong>)entry.GetField("selectedPlayers", S)!.GetValue(null)!;
            selected.Clear();
            selected.Add(me.NetId);
            var amount = (SpinBox?)entry.GetField("amount", S)!.GetValue(null);
            var resultLabel = (Label?)entry.GetField("resultLabel", S)!.GetValue(null);
            var browsers = ((System.Collections.IEnumerable)entry.GetField("browsers", S)!.GetValue(null)!).Cast<object>().ToList();
            if (amount == null || browsers.Count == 0)
            {
                Status("coop_trainer_error", new { message = "Host Tools panel not built yet" });
                return;
            }

            async Task Run(string what, Func<Task> action)
            {
                int goldBefore = me.Gold, deckBefore = me.Deck.Cards.Count, relicsBefore = me.Relics.Count;
                await Task.Delay(2500); // Host Tools refuses a second action within 2 s
                await action();
                await Task.Delay(1500);
                results.Add(new
                {
                    what,
                    message = resultLabel?.Text,
                    gold_change = me.Gold - goldBefore,
                    deck_change = me.Deck.Cards.Count - deckBefore,
                    relic_change = me.Relics.Count - relicsBefore,
                });
            }

            await Run("gold +77", () =>
            {
                amount.Value = 77;
                return (Task)entry.GetMethod("AddGold", S)!.Invoke(null, null)!;
            });

            foreach (bool cards in new[] { true, false })
            {
                object browser = browsers.First(b => (bool)b.GetType().GetField("Cards", I)!.GetValue(b)! == cards);
                Type bt = browser.GetType();
                // RefreshBrowser assigns a NEW list to Items, so read it after.
                entry.GetMethod("RefreshBrowser", S)!.Invoke(null, new[] { browser });
                var items = (List<AbstractModel>)bt.GetField("Items", I)!.GetValue(browser)!;
                List<AbstractModel> pool = items
                    .Where(m => cards || !me.Relics.Any(r => r.Id == m.Id))
                    .Where(m => !cards || _coopTrainerCard == null ||
                                string.Equals(m.Id.Entry, _coopTrainerCard, StringComparison.OrdinalIgnoreCase))
                    .ToList();
                if (pool.Count == 0)
                {
                    results.Add(new { what = cards ? "card" : "relic", message = "browser list empty" });
                    continue;
                }
                AbstractModel pick = pool[_r.Next(pool.Count)];
                bt.GetField("Selected", I)!.SetValue(browser, pick);
                if (cards && bt.GetField("Count", I)!.GetValue(browser) is SpinBox count)
                {
                    count.Value = 1;
                }
                await Run((cards ? "card " : "relic ") + pick.Id.Entry,
                    () => (Task)entry.GetMethod("AddSelected", S)!.Invoke(null, new[] { browser })!);
            }
        }
        catch (Exception ex)
        {
            Status("coop_trainer_error", new { message = (ex.InnerException ?? ex).ToString() });
        }
        var after = new { recipient = me.NetId, gold = me.Gold, deck = me.Deck.Cards.Count, relics = me.Relics.Count };
        Status("coop_trainer", new { floor = state.TotalFloor, before, after, results });
    }

    // ------------------------------------------------------------ map/rooms

    private static async Task<bool> HandleMap(NMapScreen map, RunState state)
    {
        if (!map.IsTravelEnabled || DateTime.UtcNow - _lastMapClick < TimeSpan.FromSeconds(8))
        {
            return false;
        }
        List<NMapPoint> points = UiHelper.FindAll<NMapPoint>(map)
            .Where(p => p.IsEnabled && p.IsVisibleInTree()).ToList();
        if (points.Count == 0)
        {
            return false;
        }
        // Avoid elites while anything else is open; when hurt, head for a
        // rest site, then the gentler rooms.
        Player? me = LocalContext.GetMe(state);
        bool hurt = me != null && me.Creature.CurrentHp * 2 < me.Creature.MaxHp;
        int Rank(NMapPoint p) => (p.Point.PointType, hurt) switch
        {
            (MapPointType.Elite, _) => 3,
            (MapPointType.RestSite, true) => 0,
            (MapPointType.Monster, true) => 2,
            _ => 1,
        };
        int best = points.Min(Rank);
        points = points.Where(p => Rank(p) == best).ToList();
        NMapPoint point = points[_r.Next(points.Count)];
        _lastMapClick = DateTime.UtcNow;
        Status("map_vote", new { floor = state.TotalFloor, row = point.Point.coord.row, col = point.Point.coord.col });
        await UiHelper.Click(point);
        return true;
    }

    private static async Task<bool> HandleRoom(Node root, RunState state)
    {
        int floor = state.TotalFloor;

        if (root.GetNodeOrNull(RunRoot + "EventRoom") is Node eventRoom)
        {
            List<NEventOptionButton> options = UiHelper.FindAll<NEventOptionButton>(eventRoom)
                .Where(o => o.IsVisibleInTree() && o.IsEnabled && !o.Option.IsLocked).ToList();
            if (options.Count > 0)
            {
                if (DateTime.UtcNow - _lastEventClick < TimeSpan.FromSeconds(4))
                {
                    return false;
                }
                List<NEventOptionButton> safe = options.Where(o =>
                    o.Option.WillKillPlayer == null || o.Event?.Owner == null || !o.Option.WillKillPlayer(o.Event.Owner)).ToList();
                if (safe.Count > 0)
                {
                    options = safe;
                }
                NEventOptionButton choice = options[_r.Next(options.Count)];
                _lastEventClick = DateTime.UtcNow;
                Note($"event {choice.Event?.Id.Entry}: {choice.Option.Title.GetFormattedText()}");
                await UiHelper.Click(choice);
                return true;
            }
            if (UiHelper.FindFirst<NAncientEventLayout>(eventRoom) is NAncientEventLayout ancient &&
                ancient.GetNodeOrNull<NButton>("%DialogueHitbox") is NButton hit && hit.IsVisibleInTree() && hit.IsEnabled)
            {
                Note("ancient dialogue");
                hit.EmitSignal(NClickableControl.SignalName.Released, hit);
                await Task.Delay(500);
                return true;
            }
            return await ClickProceed(eventRoom);
        }

        if (root.GetNodeOrNull<NRestSiteRoom>(RunRoot + "RestSiteRoom") is NRestSiteRoom rest)
        {
            if (await ClickProceed(rest))
            {
                return true;
            }
            if (_restChosenFloor != floor)
            {
                List<NRestSiteButton> buttons = UiHelper.FindAll<NRestSiteButton>(rest)
                    .Where(b => b.IsVisibleInTree() && b.Option.IsEnabled).ToList();
                if (buttons.Count > 0)
                {
                    // Heal below 70% HP, otherwise upgrade a card; anything
                    // else only when neither is offered.
                    Player? me = LocalContext.GetMe(state);
                    bool heal = me != null && me.Creature.CurrentHp * 10 < me.Creature.MaxHp * 7;
                    string want = heal ? "HealRestSiteOption" : "SmithRestSiteOption";
                    List<NRestSiteButton> preferred = buttons.Where(x => x.Option.GetType().Name == want).ToList();
                    if (preferred.Count == 0)
                    {
                        preferred = buttons.Where(x => x.Option.GetType().Name is "HealRestSiteOption" or "SmithRestSiteOption").ToList();
                    }
                    if (preferred.Count > 0)
                    {
                        buttons = preferred;
                    }
                    NRestSiteButton b = buttons[_r.Next(buttons.Count)];
                    _restChosenFloor = floor;
                    Note("rest site: " + b.Option.GetType().Name);
                    await UiHelper.Click(b);
                    return true;
                }
            }
            return false;
        }

        if (root.GetNodeOrNull<NTreasureRoom>(RunRoot + "TreasureRoom") is NTreasureRoom treasure)
        {
            if (_chestOpenedFloor != floor && treasure.GetNodeOrNull<NClickableControl>("Chest") is NClickableControl chest &&
                chest.IsVisibleInTree() && chest.IsEnabled)
            {
                _chestOpenedFloor = floor;
                Note("treasure: open chest");
                await UiHelper.Click(chest);
                await Task.Delay(800);
                return true;
            }
            // Once per room: in co-op a pick is a vote and the holder stays
            // clickable afterwards, so re-clicking it never moves on.
            NTreasureRoomRelicHolder? relic = UiHelper.FindAll<NTreasureRoomRelicHolder>(treasure)
                .FirstOrDefault(h => h.IsVisibleInTree() && h.IsEnabled);
            if (relic != null && _relicPickedFloor != floor && DateTime.UtcNow - _lastRoomClick > TimeSpan.FromSeconds(3))
            {
                _relicPickedFloor = floor;
                _lastRoomClick = DateTime.UtcNow;
                Note("treasure: pick relic");
                await UiHelper.Click(relic);
                return true;
            }
            return await ClickProceed(treasure);
        }

        if (root.GetNodeOrNull<NMerchantRoom>(RunRoot + "MerchantRoom") is NMerchantRoom shop)
        {
            NBackButton? back = UiHelper.FindAll<NBackButton>(shop).FirstOrDefault(b => b.IsVisibleInTree() && b.IsEnabled);
            if (back != null)
            {
                await UiHelper.Click(back);
                return true;
            }
            return await ClickProceed(shop);
        }

        // Anything else (combat room after the fight, unknown rooms): a
        // proceed button that stays available is the way on.
        if (root.GetNodeOrNull("/root/Game/RootSceneContainer/Run") is Node run &&
            DateTime.UtcNow - _lastProgress > TimeSpan.FromSeconds(6))
        {
            return await ClickProceed(run);
        }
        return false;
    }

    private static async Task<bool> ClickProceed(Node scope)
    {
        if (DateTime.UtcNow - _lastRoomClick < TimeSpan.FromSeconds(3))
        {
            return false;
        }
        NProceedButton? proceed = UiHelper.FindAll<NProceedButton>(scope)
            .FirstOrDefault(b => b.IsVisibleInTree() && b.IsEnabled);
        if (proceed == null)
        {
            return false;
        }
        _lastRoomClick = DateTime.UtcNow;
        Note("proceed in " + scope.Name);
        await UiHelper.Click(proceed);
        return true;
    }

    // -------------------------------------------------------------- helpers

    private static void Finish()
    {
        if (_done)
        {
            return;
        }
        _done = true;
        if (_followAi && _isHost)
        {
            Status("ai_follow", new { ai_moves = _aiMoves, rule_moves = _ruleMoves });
        }
        // The client quits first so the host's disconnect popup never races
        // the client's own "done".
        int delayMs = _isHost ? 8000 : 2000;
        TaskHelper.RunSafely(QuitLater(delayMs));
    }

    private static async Task QuitLater(int delayMs)
    {
        await Task.Delay(delayMs);
        Status("quit", new { });
        Callable.From(() => NGame.Instance?.GetTree().Quit(0)).CallDeferred();
    }

    private static void ReportIfStuck()
    {
        if (DateTime.UtcNow - _lastProgress < TimeSpan.FromSeconds(90) ||
            DateTime.UtcNow - _lastStuckReport < TimeSpan.FromSeconds(60))
        {
            return;
        }
        _lastStuckReport = DateTime.UtcNow;
        Status("stuck", new { idle_s = (int)(DateTime.UtcNow - _lastProgress).TotalSeconds, state = Describe() });
    }

    private static string Describe()
    {
        try
        {
            RunState? s = RunManager.Instance?.DebugOnlyGetState();
            Player? me = s == null ? null : LocalContext.GetMe(s);
            return string.Join(", ", new[]
            {
                $"in_run={RunManager.Instance?.IsInProgress}",
                $"floor={s?.TotalFloor}",
                $"room={s?.CurrentRoom?.RoomType}",
                $"overlay={(NOverlayStack.Instance?.Peek() as Node)?.GetType().Name ?? "-"}",
                $"modal={(NModalContainer.Instance?.OpenModal as Node)?.GetType().Name ?? "-"}",
                $"map_open={NMapScreen.Instance?.IsOpen}",
                $"travel={NMapScreen.Instance?.IsTravelEnabled}",
                $"combat={CombatManager.Instance?.IsInProgress}",
                $"phase={me?.PlayerCombatState?.Phase}",
                $"hand_select={NPlayerHand.Instance?.IsInCardSelection}",
                $"last={_lastNote}",
            });
        }
        catch (Exception ex)
        {
            return "describe failed: " + ex.Message;
        }
    }

    private static string CollectText(Node node)
    {
        var parts = new List<string>();
        foreach (Node n in UiHelper.FindAll<Node>(node))
        {
            string? t = n switch
            {
                MegaRichTextLabel r => r.Text,
                MegaLabel l => l.Text,
                RichTextLabel r2 => r2.Text,
                Label l2 => l2.Text,
                _ => null,
            };
            if (!string.IsNullOrWhiteSpace(t) && n is CanvasItem ci && ci.IsVisibleInTree())
            {
                parts.Add(t.Trim());
            }
        }
        string all = string.Join(" | ", parts.Distinct());
        return all.Length > 600 ? all[..600] : all;
    }

    private static void Note(string what)
    {
        _lastNote = what;
        GD.Print($"[SpireTestBot:{Role}] {what}");
    }

    private static void Status(string ev, object data)
    {
        string line = JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["t"] = DateTime.Now.ToString("HH:mm:ss"),
            ["role"] = Role,
            ["event"] = ev,
            ["data"] = data,
        });
        GD.Print($"[SpireTestBot:{Role}] {line}");
        if (_outPath == null)
        {
            return;
        }
        lock (_fileLock)
        {
            try
            {
                File.AppendAllText(_outPath, line + System.Environment.NewLine);
            }
            catch (Exception)
            {
                // A locked status file must never take the bot down.
            }
        }
    }
}
