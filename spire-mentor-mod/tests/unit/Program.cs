// Mod unit tests (no framework): each Check prints and counts; exit 1 on any failure.

using SpireMentor;

int failures = 0;
int checks = 0;

void Check(string name, string? actual, string? expected)
{
    checks++;
    if (actual == expected)
    {
        return;
    }
    failures++;
    Console.WriteLine($"FAIL {name}: got {actual ?? "<null>"}, want {expected ?? "<null>"}");
}

// GameVersionFormat.Normalize: release_info.json "version" -> wire game_version.
Check("canonical 0.111", GameVersionFormat.Normalize("v0.111.0"), "v0.111.0");
Check("canonical 0.107", GameVersionFormat.Normalize("v0.107.1"), "v0.107.1");
Check("no v prefix", GameVersionFormat.Normalize("0.111.0"), "v0.111.0");
Check("upper V + padding", GameVersionFormat.Normalize("  V0.107.1\n"), "v0.107.1");
Check("leading zeros dropped", GameVersionFormat.Normalize("v00.0111.00"), "v0.111.0");
Check("null", GameVersionFormat.Normalize(null), null);
Check("blank", GameVersionFormat.Normalize("   "), null);
// Unknown shapes are passed through (trimmed, v-prefixed), never guessed into a
// supported build: the advisor names exactly what it refuses.
Check("suffix passes through", GameVersionFormat.Normalize("v0.112.0-beta"), "v0.112.0-beta");
Check("four parts pass through", GameVersionFormat.Normalize("0.111.0.1"), "v0.111.0.1");
Check("bare v", GameVersionFormat.Normalize("v"), "v");
Check("non-ascii digits pass through", GameVersionFormat.Normalize("v0.१११.0"), "v0.१११.0");
Check("default is stable", GameVersionFormat.Default, "v0.107.1");

// ChoiceInference: which option the human picked, from a before/after diff.
// null = skip, ChoiceInference.Unknown (-1) = can't tell (send nothing).
string P(int? v) => v == null ? "skip" : v.Value.ToString();
var mapOpts = new List<(int col, int row)> { (1, 2), (3, 2), (5, 2) };
Check("map pick found", P(ChoiceInference.MapPick(mapOpts, (3, 2))), "1");
Check("map pick not offered", P(ChoiceInference.MapPick(mapOpts, (4, 2))), "-1");
Check("map pick nothing visited", P(ChoiceInference.MapPick(mapOpts, null)), "-1");
Check("newly visited: one", ChoiceInference.NewlyVisited(
    new List<(int, int)> { (0, 0), (1, 1) }, new List<(int, int)> { (0, 0), (1, 1), (3, 2) })?.ToString(), "(3, 2)");
Check("newly visited: none", ChoiceInference.NewlyVisited(
    new List<(int, int)> { (0, 0) }, new List<(int, int)> { (0, 0) })?.ToString(), null);
Check("newly visited: two is ambiguous", ChoiceInference.NewlyVisited(
    new List<(int, int)> { (0, 0) }, new List<(int, int)> { (0, 0), (1, 1), (2, 2) })?.ToString(), null);

var reward = new List<string> { "Pommel Strike", "Inflame", "Shrug It Off" };
var deck = new List<string> { "Strike", "Strike", "Defend", "Bash" };
Check("card pick", P(ChoiceInference.AddedCardPick(reward, deck, deck.Append("Inflame").ToList())), "1");
Check("card skip", P(ChoiceInference.AddedCardPick(reward, deck, deck)), "skip");
var deckWithInflame = deck.Append("Inflame").ToList();
Check("duplicate already in deck (count diff)",
    P(ChoiceInference.AddedCardPick(reward, deckWithInflame, deckWithInflame.Append("Inflame").ToList())), "1");
Check("added card not offered -> unknown",
    P(ChoiceInference.AddedCardPick(reward, deck, deck.Append("Clumsy").ToList())), "-1");
Check("offered card + unrelated add -> the offered one",
    P(ChoiceInference.AddedCardPick(reward, deck, deck.Append("Clumsy").Append("Inflame").ToList())), "1");
Check("two offered cards added -> unknown",
    P(ChoiceInference.AddedCardPick(reward, deck, deck.Append("Inflame").Append("Shrug It Off").ToList())), "-1");
Check("card removed only -> unknown",
    P(ChoiceInference.AddedCardPick(reward, deck, deck.Skip(1).ToList())), "-1");

var relics = new List<string> { "BURNING_BLOOD" };
var bossOffer = new List<string> { "SOZU", "ECTOPLASM", "PHILOSOPHERS_STONE" };
Check("relic pick", P(ChoiceInference.AddedRelicPick(bossOffer, relics, relics.Append("ECTOPLASM").ToList())), "1");
Check("relic skip", P(ChoiceInference.AddedRelicPick(bossOffer, relics, relics)), "skip");
Check("relic not offered -> unknown",
    P(ChoiceInference.AddedRelicPick(bossOffer, relics, relics.Append("ANCHOR").ToList())), "-1");

// Same rule as SimAdvisor.ReconcileIndex: trimmed, case-insensitive, first match.
var labels = new List<string> { "Strike", " inflame ", "Inflame" };
Check("label exact", ChoiceInference.IndexByLabel(labels, "Strike")?.ToString(), "0");
Check("label case/trim, first match wins", ChoiceInference.IndexByLabel(labels, "INFLAME")?.ToString(), "1");
Check("label missing", ChoiceInference.IndexByLabel(labels, "Bash")?.ToString(), null);
Check("label empty list", ChoiceInference.IndexByLabel(new List<string>(), "Bash")?.ToString(), null);
Check("same label twice in an offer -> first",
    P(ChoiceInference.AddedCardPick(new List<string> { "Inflame", "Inflame" }, deck, deck.Append("Inflame").ToList())), "0");

var bundles = new List<IReadOnlyList<string>>
{
    new List<string> { "A", "B" },
    new List<string> { "C", "D" },
};
Check("bundle pick", P(ChoiceInference.BundlePick(bundles, deck, deck.Append("D").Append("C").ToList())), "1");
Check("bundle partial -> unknown", P(ChoiceInference.BundlePick(bundles, deck, deck.Append("C").ToList())), "-1");
Check("bundle nothing -> unknown (no skip)", P(ChoiceInference.BundlePick(bundles, deck, deck)), "-1");

var shop = new List<(string action, string id)>
{
    ("leave_shop", "leave_shop"), ("buy_card", "Inflame"), ("buy_relic", "ANCHOR"),
    ("buy_potion", "FIRE_POTION"), ("remove_card", "remove_card"),
};
var noPotions = new List<string>();
Check("shop buys card", P(ChoiceInference.ShopPick(shop, deck, deck.Append("Inflame").ToList(),
    relics, relics, noPotions, noPotions)), "1");
Check("shop buys relic", P(ChoiceInference.ShopPick(shop, deck, deck,
    relics, relics.Append("ANCHOR").ToList(), noPotions, noPotions)), "2");
Check("shop buys potion", P(ChoiceInference.ShopPick(shop, deck, deck,
    relics, relics, noPotions, new List<string> { "FIRE_POTION" })), "3");
Check("shop removes a card", P(ChoiceInference.ShopPick(shop, deck, deck.Skip(1).ToList(),
    relics, relics, noPotions, noPotions)), "4");
Check("shop nothing -> unknown", P(ChoiceInference.ShopPick(shop, deck, deck,
    relics, relics, noPotions, noPotions)), "-1");
Check("shop relic + card -> unknown", P(ChoiceInference.ShopPick(shop, deck, deck.Append("Inflame").ToList(),
    relics, relics.Append("ANCHOR").ToList(), noPotions, noPotions)), "-1");

Check("single changed", P(ChoiceInference.SingleChanged(new List<bool> { false, true, false })), "1");
Check("none changed", P(ChoiceInference.SingleChanged(new List<bool> { false, false })), "-1");
Check("two changed", P(ChoiceInference.SingleChanged(new List<bool> { true, true })), "-1");

// CoachStyle: F8 toggles the AI on/off and keeps the style; the style button
// flips Hints <-> Coach and never turns the AI off.
string T(SimModeLite m, AdviceStyle s) => CoachStyle.ToggleOnOff(m, s).ToString();
string W(SimModeLite m, AdviceStyle s) => CoachStyle.SwitchStyle(m, s).ToString();
Check("toggle off -> on keeps coach", T(SimModeLite.Off, AdviceStyle.Coach), "(Advisor, Coach)");
Check("toggle off -> on keeps hints", T(SimModeLite.Off, AdviceStyle.Hints), "(Advisor, Hints)");
Check("toggle on -> off", T(SimModeLite.Advisor, AdviceStyle.Coach), "(Off, Coach)");
Check("toggle autopilot -> advisor", T(SimModeLite.Autopilot, AdviceStyle.Coach), "(Advisor, Coach)");
Check("style coach -> hints stays on", W(SimModeLite.Advisor, AdviceStyle.Coach), "(Advisor, Hints)");
Check("style hints -> coach stays on", W(SimModeLite.Advisor, AdviceStyle.Hints), "(Advisor, Coach)");
Check("style while off stays off", W(SimModeLite.Off, AdviceStyle.Hints), "(Off, Coach)");
Check("power label on", CoachStyle.PowerLabel(SimModeLite.Advisor), "AI on");
Check("power label off", CoachStyle.PowerLabel(SimModeLite.Off), "AI off");
Check("power label autopilot", CoachStyle.PowerLabel(SimModeLite.Autopilot), "Take over");
Check("style label hints", CoachStyle.StyleLabel(AdviceStyle.Hints), "Hints");
Check("style label coach", CoachStyle.StyleLabel(AdviceStyle.Coach), "Coach");
Check("mode name coach", CoachStyle.ModeName(SimModeLite.Advisor, AdviceStyle.Coach), "Coach");
Check("mode name autopilot", CoachStyle.ModeName(SimModeLite.Autopilot, AdviceStyle.Coach), "Autopilot");
Check("fresh config default is Coach",
    CoachStyle.FromDefaultFlag(CoachStyle.CoachByDefault).ToString(), "Coach");
Check("default flag off -> Hints", CoachStyle.FromDefaultFlag(false).ToString(), "Hints");
Check("parse hints", CoachStyle.Parse(" HINTS ")?.ToString(), "Hints");
Check("parse coach", CoachStyle.Parse("coach")?.ToString(), "Coach");
Check("parse junk", CoachStyle.Parse("advisor")?.ToString(), null);

// CoachChipText.Format: one fixture per row of the plan's copy table.
string Chip(string json, bool showGood = false)
{
    var obj = (System.Text.Json.Nodes.JsonObject)System.Text.Json.Nodes.JsonNode.Parse(json)!;
    ChipText? c = CoachChipText.Format(obj, showGood);
    return c == null ? "<none>" : $"[{c.Tone}] {c.Title} | {string.Join(" | ", c.Lines)}";
}
const string turnBase = "\"kind\":\"turn\",\"id\":\"turn:n4\",\"node_id\":\"n4\",\"turn\":3,\"floor\":5," +
    "\"you\":[\"Strike → Cultist\",\"Strike → Cultist\",\"end turn\"],\"ai\":[\"Bash → Cultist\",\"Defend\",\"end turn\"]," +
    "\"ungraded_actions\":0";
Check("turn blunder with HP",
    Chip("{" + turnBase + ",\"verdict\":\"blunder\",\"hp_drop\":9.4,\"win_before\":0.96,\"win_after\":0.941,\"peeked\":false}"),
    "[Blunder] ⚠ blunder | Turn 3 · you: Strike → Cultist → Strike → Cultist → end turn" +
    " | AI:  Bash → Cultist → Defend → end turn | ≈ 9 HP more lost · win 96% → 94%");
Check("turn blunder with a reason line",
    Chip("{" + turnBase + ",\"verdict\":\"blunder\",\"hp_drop\":9.4,\"win_before\":0.96,\"win_after\":0.941,\"peeked\":false," +
         "\"why\":\"Your line wakes Lagavulin Matriarch on turn 3\"}"),
    "[Blunder] ⚠ blunder | Turn 3 · you: Strike → Cultist → Strike → Cultist → end turn" +
    " | AI:  Bash → Cultist → Defend → end turn | ≈ 9 HP more lost · win 96% → 94%" +
    " | Your line wakes Lagavulin Matriarch on turn 3");
Check("turn slip without HP",
    Chip("{" + turnBase + ",\"verdict\":\"slip\",\"hp_drop\":null,\"win_before\":0.80,\"win_after\":0.75,\"peeked\":false}"),
    "[Slip] ▽ suboptimal | Turn 3 · you: Strike → Cultist → Strike → Cultist → end turn" +
    " | AI:  Bash → Cultist → Defend → end turn | win 80% → 75%");
Check("turn slip, HP rounds below 1 -> dropped",
    Chip("{" + turnBase + ",\"verdict\":\"slip\",\"hp_drop\":0.4,\"win_before\":0.80,\"win_after\":0.75,\"peeked\":false}"),
    "[Slip] ▽ suboptimal | Turn 3 · you: Strike → Cultist → Strike → Cultist → end turn" +
    " | AI:  Bash → Cultist → Defend → end turn | win 80% → 75%");
Check("turn peeked suffix",
    Chip("{" + turnBase + ",\"verdict\":\"slip\",\"hp_drop\":4,\"win_before\":0.9,\"win_after\":0.9,\"peeked\":true}"),
    "[Slip] ▽ suboptimal | Turn 3 · you: Strike → Cultist → Strike → Cultist → end turn" +
    " | AI:  Bash → Cultist → Defend → end turn | ≈ 4 HP more lost · win 90% → 90% (peeked)");
Check("turn tie hidden",
    Chip("{" + turnBase + ",\"verdict\":\"tie\",\"hp_drop\":1,\"win_before\":0.9,\"win_after\":0.9,\"peeked\":false}"),
    "<none>");
Check("turn tie shown",
    Chip("{" + turnBase + ",\"verdict\":\"tie\",\"hp_drop\":1,\"win_before\":0.9,\"win_after\":0.9,\"peeked\":false}", showGood: true),
    "[Good] ✓ Turn 3 · toss-up | ");
Check("turn best shown",
    Chip("{" + turnBase + ",\"verdict\":\"best\",\"hp_drop\":0,\"win_before\":0.9,\"win_after\":0.9,\"peeked\":false}", showGood: true),
    "[Good] ✓ Turn 3 | ");
Check("turn best hidden",
    Chip("{" + turnBase + ",\"verdict\":\"best\",\"hp_drop\":0,\"win_before\":0.9,\"win_after\":0.9,\"peeked\":false}"),
    "<none>");

string ChoiceJson(int rank, string extra = "", string verdict = "flag") =>
    "{\"kind\":\"choice\",\"id\":\"choice:n9\",\"node_id\":\"n9\",\"decision\":\"card_reward\"," +
    "\"label_kind\":\"Card reward\",\"verdict\":\"" + verdict + "\",\"strength\":\"strong preference\"," +
    "\"played\":{\"label\":\"Skip\",\"prob\":0.05,\"rank\":" + rank + "},\"top\":{\"label\":\"Inflame\",\"prob\":0.72}," +
    "\"alternatives\":[],\"peeked\":false" + extra + "}";
Check("choice flag rank 3", Chip(ChoiceJson(3)),
    "[Info] Card reward · you: Skip | AI would take: Inflame (strong preference) · Skip was its 3rd choice");
Check("choice flag rank 11", Chip(ChoiceJson(11)),
    "[Info] Card reward · you: Skip | AI would take: Inflame (strong preference) · Skip was its 11th choice");
Check("choice flag rank 2 / 22", Chip(ChoiceJson(22)),
    "[Info] Card reward · you: Skip | AI would take: Inflame (strong preference) · Skip was its 22nd choice");
Check("choice agree hidden", Chip(ChoiceJson(1, verdict: "agree")), "<none>");
Check("choice agree shown", Chip(ChoiceJson(1, verdict: "agree"), showGood: true), "[Good] ✓ AI agrees | ");
Check("choice tossup shown", Chip(ChoiceJson(2, verdict: "tossup"), showGood: true), "[Good] ✓ toss-up | ");
const string route = ",\"path\":{\"you\":[{\"col\":1,\"row\":2,\"hop\":0,\"hp\":60},{\"col\":1,\"row\":3,\"hop\":1,\"hp\":51.6}]," +
    "\"ai\":[{\"col\":3,\"row\":2,\"hop\":0,\"hp\":60},{\"col\":3,\"row\":3,\"hop\":1,\"hp\":58.2}]}";
Check("map route line present", Chip(ChoiceJson(2, route).Replace("Card reward", "Map")),
    "[Info] Map · you: Skip | AI would take: Inflame (strong preference) · Skip was its 2nd choice | route HP: you ≈52 · AI ≈58");
const string halfRoute = ",\"path\":{\"you\":[{\"col\":1,\"row\":2,\"hop\":0}],\"ai\":[{\"col\":3,\"row\":2,\"hop\":0,\"hp\":60}]}";
Check("map route line absent without both HPs", Chip(ChoiceJson(2, halfRoute).Replace("Card reward", "Map")),
    "[Info] Map · you: Skip | AI would take: Inflame (strong preference) · Skip was its 2nd choice");
Check("withdraw is not text", Chip("{\"kind\":\"withdraw\",\"ids\":[\"turn:n4\"]}"), "<none>");
Check("unknown kind", Chip("{\"kind\":\"recap\",\"id\":\"x\"}"), "<none>");
Check("missing kind", Chip("{\"id\":\"x\"}"), "<none>");
Check("ordinal 1/12/13/101/111", string.Join(",", new[] { 1, 12, 13, 101, 111, 112, 23 }.Select(CoachChipText.Ordinal)),
    "1st,12th,13th,101st,111th,112th,23rd");

// ChipText carries the record's run-log position (v5.1).
ChipText? Raw(string json) => CoachChipText.Format(
    (System.Text.Json.Nodes.JsonObject)System.Text.Json.Nodes.JsonNode.Parse(json)!, showGood: false);
ChipText? located = Raw(ChoiceJson(3, ",\"log_name\":\"run_a.jsonl\",\"log_idx\":7"));
Check("chip carries log position", $"{located?.LogName}@{located?.LogIdx}", "run_a.jsonl@7");
ChipText? unlocated = Raw(ChoiceJson(3, ",\"log_name\":null,\"log_idx\":null"));
Check("chip without log position", $"{unlocated?.LogName ?? "-"}@{unlocated?.LogIdx?.ToString() ?? "-"}", "-@-");

// WorkbenchUrl.WithFollowParams: the "open in workbench" link.
Check("url: bundled with run+step",
    WorkbenchUrl.WithFollowParams("http://127.0.0.1:8079/?follow=1", "x y.jsonl", 3),
    "http://127.0.0.1:8079/?follow=1&run=x%20y.jsonl&step=3");
Check("url: configured without a query",
    WorkbenchUrl.WithFollowParams("http://localhost:5173/", "a.jsonl", 0),
    "http://localhost:5173/?follow=1&run=a.jsonl&step=0");
Check("url: step only", WorkbenchUrl.WithFollowParams("http://h/?follow=1", null, 12),
    "http://h/?follow=1&step=12");
Check("url: no params unchanged", WorkbenchUrl.WithFollowParams("http://h/?follow=1", null, null),
    "http://h/?follow=1");
Check("url: other query keeps it", WorkbenchUrl.WithFollowParams("http://h/?x=1#top", "r", 2),
    "http://h/?x=1&follow=1&run=r&step=2#top");

// CoachPanelText.Format: recap / autopsy panels (plan 4 copy table).
string Panel(string json)
{
    var obj = (System.Text.Json.Nodes.JsonObject)System.Text.Json.Nodes.JsonNode.Parse(json)!;
    PanelText? p = CoachPanelText.Format(obj);
    if (p == null)
    {
        return "<none>";
    }
    var parts = new List<string> { p.Title, p.Summary };
    foreach (PanelSection s in p.Sections)
    {
        parts.Add($"[{s.Heading}{(s.LogIdx is int i ? "@" + i : "")}] " + string.Join(" / ", s.Lines));
    }
    if (p.Footer.Length > 0)
    {
        parts.Add(p.Footer);
    }
    return string.Join(" | ", parts);
}
const string ctx = "\"context\":{\"hand\":[\"Bash\",\"Strike\",\"Defend\"],\"energy\":3,\"max_energy\":3,\"hp\":70,\"max_hp\":80,\"block\":0," +
    "\"enemies\":[{\"name\":\"Jaw Worm\",\"hp\":40,\"intent\":\"attack 11\"}]}";
string CostlyTurn(string extra) =>
    "{\"kind\":\"turn\",\"id\":\"turn:n7\",\"node_id\":\"n7\",\"turn\":2,\"floor\":5,\"verdict\":\"blunder\"," +
    "\"you\":[\"Strike\",\"end turn\"],\"ai\":[\"Bash\",\"Defend\",\"end turn\"],\"win_before\":0.97,\"win_after\":0.95," +
    "\"severity\":9,\"log_name\":\"r.jsonl\",\"log_idx\":12," + extra + "}";
string RecapJson(string extra, string costliest = "") =>
    "{\"kind\":\"recap\",\"id\":\"recap:n5\",\"node_id\":\"n5\",\"log_name\":\"r.jsonl\",\"log_idx\":10,\"floor\":5," +
    "\"room_type\":\"Monster\",\"hp_start\":80,\"hp_end\":76,\"hp_lost\":4,\"turns\":3,\"turns_graded\":3,\"unfinished_turns\":1," +
    "\"costliest\":[" + costliest + "]," + extra + "}";
Check("recap clean", Panel(RecapJson("\"ai_hp_lost\":3.2,\"turns_ungraded\":0")),
    "Fight recap · floor 5 | −4 HP (AI line typically −3)");
Check("recap without HP samples", Panel(RecapJson("\"ai_hp_lost\":null,\"turns_ungraded\":0")),
    "Fight recap · floor 5 | −4 HP");
Check("recap with ungraded turns", Panel(RecapJson("\"ai_hp_lost\":null,\"turns_ungraded\":2")),
    "Fight recap · floor 5 | −4 HP · 2 turns ungraded");
Check("recap one costliest turn",
    Panel(RecapJson("\"ai_hp_lost\":6,\"turns_ungraded\":1", CostlyTurn(ctx + ",\"hp_drop\":5.4,\"peeked\":false"))),
    "Fight recap · floor 5 | −4 HP (AI line typically −6) · 1 turn ungraded" +
    " | [Turn 2 — hand: Bash, Strike, Defend · 3 energy@12] Jaw Worm intends: attack 11 / You: Strike → end turn" +
    " / AI:  Bash → Defend → end turn / ≈ 5 HP more lost");
Check("recap costliest turn with a reason line",
    Panel(RecapJson("\"ai_hp_lost\":null,\"turns_ungraded\":0",
        CostlyTurn(ctx + ",\"hp_drop\":5.4,\"peeked\":false,\"why\":\"Your line wakes Jaw Worm on turn 2\""))),
    "Fight recap · floor 5 | −4 HP" +
    " | [Turn 2 — hand: Bash, Strike, Defend · 3 energy@12] Jaw Worm intends: attack 11 / You: Strike → end turn" +
    " / AI:  Bash → Defend → end turn / ≈ 5 HP more lost / Your line wakes Jaw Worm on turn 2");
Check("recap costliest without hp_drop -> win line, peeked",
    Panel(RecapJson("\"ai_hp_lost\":null,\"turns_ungraded\":0", CostlyTurn(ctx + ",\"hp_drop\":null,\"peeked\":true"))),
    "Fight recap · floor 5 | −4 HP" +
    " | [Turn 2 — hand: Bash, Strike, Defend · 3 energy@12] Jaw Worm intends: attack 11 / You: Strike → end turn" +
    " / AI:  Bash → Defend → end turn / win 97% → 95% (peeked)");
Check("autopsy with turn and choice",
    Panel("{\"kind\":\"autopsy\",\"id\":\"autopsy:S\",\"log_name\":\"r.jsonl\",\"floor\":9,\"act\":1,\"items\":[" +
          "{\"kind\":\"turn\",\"record\":" + CostlyTurn(ctx + ",\"hp_drop\":8,\"peeked\":false") + "}," +
          "{\"kind\":\"choice\",\"record\":" + ChoiceJson(3, ",\"log_name\":\"r.jsonl\",\"log_idx\":4") + "}]," +
          "\"turns_ungraded\":1,\"decisions_graded\":14}"),
    "Run autopsy · died on floor 9 | " +
    " | [1. Combat turn@12] Floor 5 · Turn 2 — hand: Bash, Strike, Defend · 3 energy / Jaw Worm intends: attack 11" +
    " / You: Strike → end turn / AI:  Bash → Defend → end turn / ≈ 8 HP more lost" +
    " | [2. Card reward@4] Card reward · you: Skip / AI would take: Inflame (strong preference) · Skip was its 3rd choice" +
    " | 14 decisions graded · 1 turn ungraded");
Check("autopsy empty",
    Panel("{\"kind\":\"autopsy\",\"id\":\"autopsy:S\",\"log_name\":null,\"floor\":3,\"act\":1,\"items\":[],\"turns_ungraded\":0,\"decisions_graded\":2}"),
    "Run autopsy · died on floor 3 | No costly decisions found. | 2 decisions graded · 0 turns ungraded");
Check("panel unknown kind", Panel("{\"kind\":\"turn\",\"id\":\"x\"}"), "<none>");
var recapPanel = CoachPanelText.Format((System.Text.Json.Nodes.JsonObject)System.Text.Json.Nodes.JsonNode.Parse(
    RecapJson("\"ai_hp_lost\":null,\"turns_ungraded\":0", CostlyTurn(ctx + ",\"hp_drop\":5,\"peeked\":false")))!);
Check("panel carries log name and section step", $"{recapPanel?.LogName}@{recapPanel?.Sections[0].LogIdx}", "r.jsonl@12");

// SimCoachReports: payloads and the once-per-run game_over latch.
string Payload(Dictionary<string, object> d) => System.Text.Json.JsonSerializer.Serialize(d);
Check("combat_ended payload", Payload(SimCoachReports.CombatEnded(true, 4, 61, 80, "Monster")),
    "{\"type\":\"combat_ended\",\"victory\":true,\"round\":4,\"hp\":61,\"max_hp\":80,\"room_type\":\"Monster\"}");
Check("game_over payload", Payload(SimCoachReports.GameOver(false)),
    "{\"type\":\"game_over\",\"result\":\"defeat\",\"message\":\"Game over\",\"victory\":false}");
var runA = new object();
var runB = new object();
Check("latch: first claim", SimCoachReports.TryClaimGameOver(runA).ToString(), "True");
Check("latch: second claim same run", SimCoachReports.TryClaimGameOver(runA).ToString(), "False");
Check("latch: a new run claims again", SimCoachReports.TryClaimGameOver(runB).ToString(), "True");
Check("latch: no run never claims", SimCoachReports.TryClaimGameOver(null).ToString(), "False");

// AutoplayPause: `pause_on` parsing and where F7 hands back (GH #874).
string PauseOn(string confJson)
{
    using var doc = System.Text.Json.JsonDocument.Parse(confJson);
    return string.Join(",", AutoplayPause.FromJson(doc.RootElement, "pause_on", AutoplayPause.Default));
}
Check("pause_on absent -> unsupported", PauseOn("{}"), "unsupported");
Check("pause_on string list", PauseOn("{\"pause_on\":\"unsupported, Neow\"}"), "unsupported,neow");
Check("pause_on array", PauseOn("{\"pause_on\":[\"Unsupported\",\"CRYSTAL_SPHERE\"]}"), "unsupported,crystalsphere");
Check("pause_on empty string -> never", PauseOn("{\"pause_on\":\"\"}"), "");
Check("pause_on blanks and dupes", PauseOn("{\"pause_on\":\" , neow,Neow ,\"}"), "neow");
Check("pause_on null -> never", PauseOn("{\"pause_on\":null}"), "");
Check("pause_on wrong type -> default", PauseOn("{\"pause_on\":3}"), "unsupported");
Check("advisor_autoplay forces never",
    AutoplayPause.Effective(AutoplayPause.Parse("unsupported, neow"), advisorAutoplay: true).Count.ToString(), "0");
Check("attended keeps the list",
    string.Join(",", AutoplayPause.Effective(AutoplayPause.Parse("unsupported"), advisorAutoplay: false)), "unsupported");

IReadOnlyList<string> unsupportedOnly = AutoplayPause.Parse("unsupported");
Check("hand_back pauses under unsupported",
    AutoplayPause.Reason(unsupportedOnly, "card bundles have no trained policy", new[] { "card_bundle" }),
    "not covered by the AI (card bundles have no trained policy)");
Check("a real decision plays on",
    AutoplayPause.Reason(unsupportedOnly, null, new[] { "event", "NEventRoom", "NEOW" }), null);
Check("hand_back ignored without unsupported",
    AutoplayPause.Reason(AutoplayPause.Parse("Neow"), "no head", new[] { "shop" }), null);
Check("never pause",
    AutoplayPause.Reason(Array.Empty<string>(), "no head", new[] { "NEOW" }), null);
Check("a listed event id pauses",
    AutoplayPause.Reason(AutoplayPause.Parse("unsupported, Neow"), null, new[] { "event", "NEventRoom", "NEOW" }),
    "you asked to play this yourself (pause_on)");
Check("a listed advice_for pauses",
    AutoplayPause.Reason(AutoplayPause.Parse("shop"), null, new[] { "shop", "NMerchantRoom", null }),
    "you asked to play this yourself (pause_on)");
Check("an unlisted event plays on",
    AutoplayPause.Reason(AutoplayPause.Parse("Neow"), null, new[] { "event", "NEventRoom", "CRYSTAL_SPHERE" }), null);
Check("pretty event id", AutoplayPause.Pretty("CRYSTAL_SPHERE"), "Crystal Sphere");
Check("pretty advice_for", AutoplayPause.Pretty("card_bundle"), "Card Bundle");
Check("pretty nothing", AutoplayPause.Pretty(null), "This decision");
Check("toast", AutoplayPause.Toast("Crystal Sphere", "not covered by the AI (x)"),
    "Crystal Sphere: not covered by the AI (x) — your call. Ctrl+Shift+P to resume auto mode.");

// EnemyTagText: the "#k" beside a same-named enemy, numbered as the advisor's
// "→ Name #k" labels are (sts2-pycli/python/enemy_labels.py).
string TagList(params string?[] names) =>
    string.Join(",", EnemyTagText.Tags(names).Select(t => t ?? "-"));
Check("tags: three rats", TagList("Rat", "Rat", "Rat"), "#1,#2,#3");
Check("tags: unique names stay bare", TagList("Cultist", "Jaw Worm"), "-,-");
Check("tags: mixed, ranked within each name", TagList("Louse", "Cultist", "Louse"), "#1,-,#2");
Check("tags: a dead enemy (null) is not counted", TagList("Rat", null, "Rat"), "#1,-,#2");
Check("tags: one live rat beside a dead one stays bare", TagList("Rat", null), "-,-");
Check("tags: empty", TagList(), "");

Console.WriteLine(failures == 0
    ? $"ModUnitTests: OK ({checks} checks)"
    : $"ModUnitTests: {failures} of {checks} checks FAILED");
return failures == 0 ? 0 : 1;
