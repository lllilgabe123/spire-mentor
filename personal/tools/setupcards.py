"""Which beta cards make the plays after them better this turn ("setup"),
and what they help: attack / block / skill / power / any.

From the decompiled v0.111.0 card code (the powers each card's OnPlay
applies, and on whom) plus the game's power text. Output setupcards.json:
{CARD_ID: category}. usage: python setupcards.py OUT.json"""
import json
import os
import re
import sys

D = os.path.join(os.environ["LOCALAPPDATA"], "Temp", "claude-sts2-decomp", "MegaCrit.Sts2.Core.Models")
LOC = os.path.join(os.environ["LOCALAPPDATA"], "sm073", "sts2-pycli", "localization_eng", "powers.json")
power_text = {k[:-12]: re.sub(r"\[/?[a-z]+\]|\{[^}]*\}", "#", v)
              for k, v in json.load(open(LOC, encoding="utf-8")).items() if k.endswith(".description")}


def snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


SELF_BUFF = {"STRENGTH_POWER": "attack", "VIGOR_POWER": "attack", "DEXTERITY_POWER": "block",
             "FOCUS_POWER": "any"}
ENEMY_DEBUFF = {"VULNERABLE_POWER": "attack"}
#: Hand setup no power text shows: Armaments upgrades cards in hand,
#: Enlightenment makes them cheaper.
MANUAL = {"ARMAMENTS": "any", "ENLIGHTENMENT": "any"}


def per_play(text):
    """What a 'whenever you play ...' power pays off on, or None."""
    m = re.search(r"[Ww]henever you play (a card|an? Attack|a Skill|a Power)(?! that costs)", text)
    if not m:
        if re.search(r"next (Attack|Skill)s? .*(extra time|twice|double)|(Attack|Skill)s? (is|are) played an extra time", text):
            return "attack" if "Attack" in text else "skill"
        return None
    w = m.group(1)
    return "any" if w == "a card" else "attack" if "Attack" in w else "skill" if "Skill" in w else "power"


pools = {}
for p in ["Ironclad", "Silent", "Defect", "Necrobinder", "Regent", "Colorless"]:
    src = open(f"{D}.CardPools/{p}CardPool.cs", encoding="utf-8").read()
    for n in re.findall(r"<([A-Z][A-Za-z]+)>\(\)", src):
        pools.setdefault(n, p)

out = {}
for name in pools:
    try:
        s = open(f"{D}.Cards/{name}.cs", encoding="utf-8").read()
    except OSError:
        continue
    cid = snake(name)
    m = re.search(r":\s*base\((-?\d+),\s*CardType\.(\w+)", s)
    if not m:
        continue
    ctype = m.group(2)
    play = s[s.find("OnPlay"):] if "OnPlay" in s else ""
    nxt = re.search(r"\n\t(public|protected|private)[^\n]*\(", play[10:])
    if nxt:
        play = play[: 10 + nxt.start()]
    # Only effects that pay off on LATER plays THIS turn. Tested on 60 logged
    # turns, "every Power first" changed 11 first cards: 1 better, 6 worse
    # (Neurosurge, Friendship, Danse Macabre: start-of-turn or conditional
    # effects, where moving them first only spends the planned draw/energy).
    cat = MANUAL.get(cid)
    for pw, target, amount in re.findall(r"PowerCmd\.Apply<([A-Za-z]+)>\([a-zA-Z]+,\s*([^,]+),\s*([^,]+),", play):
        key = snake(pw)
        if amount.strip().startswith("-"):
            continue  # a loss (Friendship and Shared Fate cost Strength): never setup
        text = power_text.get(key.removesuffix("_POWER") + "_POWER", "") or power_text.get(key, "")
        on_self = "Owner.Creature" in target
        on_enemy = "Target" in target or "Enem" in target
        found = None
        if on_self and key in SELF_BUFF:
            found = SELF_BUFF[key]
        elif on_enemy and key in ENEMY_DEBUFF:
            found = ENEMY_DEBUFF[key]
        elif per_play(text):
            found = per_play(text)
        if found:
            cat = "any" if cat and cat != found else found if cat is None else cat
    if cat:
        out[cid] = cat

json.dump(out, open(sys.argv[1], "w"), indent=0, sort_keys=True)
by = {}
for k, v in out.items():
    by.setdefault(v, []).append(k)
for k, v in sorted(by.items()):
    print(f"{k} ({len(v)}): {', '.join(sorted(v))}")
