"""Per-card loop data from the decompiled beta: cost, energy gained, cards
drawn, cards discarded, exhaust, display name, breaks-a-loop. Output: loopcards.json {norm name: [...]}."""
import json, os, re, sys
D = os.path.join(os.environ["LOCALAPPDATA"], "Temp", "claude-sts2-decomp", "MegaCrit.Sts2.Core.Models")
pools = {}
for p in ["Ironclad", "Silent", "Defect", "Necrobinder", "Regent", "Colorless"]:
    src = open(f"{D}.CardPools/{p}CardPool.cs", encoding="utf-8").read()
    for n in re.findall(r"<([A-Z][A-Za-z]+)>\(\)", src):
        pools.setdefault(n, p.lower())
out = {}
for name, pool in pools.items():
    try:
        s = open(f"{D}.Cards/{name}.cs", encoding="utf-8").read()
    except OSError:
        continue
    m = re.search(r":\s*base\((-?\d+),\s*CardType\.(\w+)", s)
    if not m:
        continue
    cost, ctype = int(m.group(1)), m.group(2)
    xcost = "HasEnergyCostX => true" in s
    def var(kind):
        v = re.search(kind + r"\((\d+)m?\)", s)
        return int(v.group(1)) if v else 0
    # The OnPlay method only: later hooks (AfterCardExhausted on Drum of
    # Battle) are not part of playing the card.
    play = s[s.find("OnPlay"):] if "OnPlay" in s else ""
    nxt = re.search(r"\n\t(public|protected|private)[^\n]*\(", play[10:])
    if nxt:
        play = play[: 10 + nxt.start()]
    energy = var("new EnergyVar") if "GainEnergy" in play else 0
    draw = var("new CardsVar") if "CardPileCmd.Draw(" in play and "DynamicVars.Cards" in play else 0
    if "CardPileCmd.Draw(choiceContext, 1m" in play:
        draw = max(draw, 1)
    discard = 1 if "FromHandForDiscard" in play else 0
    kw = s.split("CanonicalKeywords", 1)[1].split(";", 1)[0] if "CanonicalKeywords" in s else ""
    exhaust = "CardKeyword.Exhaust" in kw
    cond = bool(re.search(r"if \(|IsPlayable", play))
    # Breaks a loop: stops further draws this turn (Battle Trance, Bullet
    # Time), adds junk cards (Overclock's Burn), or raises costs (Borrowed Time).
    noloop = bool(re.search(r"NoDrawPower|<(Burn|Wound|Dazed|Slimed|Void|Infection)>|BorrowedTimePower", s))
    out[name.lower()] = [pool, cost, ctype, xcost, energy, draw, discard, exhaust, cond, re.sub(r"(?<!^)(?=[A-Z])", " ", name), noloop]
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
print(len(out))
