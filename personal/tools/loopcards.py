"""Per-card loop data from the decompiled beta: cost, energy gained, cards
drawn, cards discarded, exhaust. Output: loopcards.json {norm name: [...]}."""
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
    play = s[s.find("OnPlay"):] if "OnPlay" in s else ""
    energy = var("new EnergyVar") if "GainEnergy" in play else 0
    draw = var("new CardsVar") if "CardPileCmd.Draw(" in play and "DynamicVars.Cards" in play else 0
    if "CardPileCmd.Draw(choiceContext, 1m" in play:
        draw = max(draw, 1)
    discard = 1 if "FromHandForDiscard" in play else 0
    exhaust = "CardKeyword.Exhaust)" in s.split("CanonicalKeywords")[1][:300] if "CanonicalKeywords" in s else False
    cond = bool(re.search(r"if \(|IsPlayable", play))
    out[name.lower()] = [pool, cost, ctype, xcost, energy, draw, discard, exhaust, cond, re.sub(r"(?<!^)(?=[A-Z])", " ", name)]
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
print(len(out))
