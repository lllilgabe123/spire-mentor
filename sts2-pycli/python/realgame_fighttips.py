"""Personal-build fight tips: how to play each beta elite and boss.

Display only. Written from each monster's move code in the decompiled
v0.111.0 game (moves, powers, status cards, A10 numbers) and the game's own
power text, not from web guides: the guide sites checked mixed in Slay the
Spire 1 bosses (Hexaghost, Slime Boss) and none listed the beta's elites.
The advice's move ranking already plays these mechanics; the tips say WHY,
and are shown on the map for the next elite/boss and in the fight itself.
"""

from __future__ import annotations

import re
from typing import Any

#: normalized monster name -> tip. Encounters list their monsters below.
TIPS: dict[str, str] = {
    # ---- elites ----
    "bygoneeffigy": "Asleep at first, wakes with +10 Strength and hits 15. Slow: every card you play makes it take "
                    "10% more Attack damage that turn, so play your Skills/Powers first and your big Attacks last",
    "byrdonis": "Territorial: +1 Strength every turn, forever. Race it: damage now beats Block early",
    "decimillipedesegment": "Segments come back with 25 HP 2 turns after dying while another lives. "
                            "Spread damage and finish them close together; AoE is gold here",
    "entomancer": "Personal Hive: every time you hit it with an Attack, a Dazed goes into your draw pile. "
                  "Few big hits beat many small ones; Bees (3x7-8) then Spear (18-20)",
    "infestedprism": "Vital Spark: your Skills are Tainted. Lean on Attacks; Whirlwind 3x5-6 punishes low Block",
    "spectralknight": "Hex: while Spectral Knight lives, ALL your cards are Ethereal (lost at end of turn). Kill it first",
    "flailknight": "War Chant gives +3 Strength; Ram hits 15-17. Kill order: Spectral Knight, then this one",
    "magiknight": "Prep (Block) is followed by Magic Bomb for 35-40: have big Block ready the turn after Prep",
    "mechaknight": "300+ HP, Artifact 3 (your first 3 debuffs are blocked). Flamethrower puts 4 Burns in your hand; "
                   "Windup (Block + 5 Strength) means Heavy Cleave for 35-40+ next turn: block it",
    "phantasmalgardener": "Skittish: the first hit each turn gives it Block, so open with your smallest hit. "
                          "Enlarge adds Strength: kill the ones that buffed first",
    "phrogparasite": "Infect adds 3 Infections (damage if held at end of turn: play or exhaust them). "
                     "Infested: when it dies, Wrigglers come out, so keep some damage for after",
    "skulkingcolony": "Hardened Shell: it can't lose more than 20 HP a turn. Don't overkill: deal ~20, Block the rest. "
                      "Inertia adds Strength over time",
    "soulnexus": "Drain Life puts Vulnerable and Weak on you; Maelstrom is a multi-hit. Block the multi-hit turns",
    "terroreel": "Shriek: the first time it drops to half HP it is stunned for a turn, then Terror gives you "
                 "99 Vulnerable. Cross half HP only when you can finish it or block hard after",
    # ---- bosses ----
    "aeonglass": "500+ HP, Artifact 3, ramps Strength every Increasing Intensity and adds Withers (damage if held). "
                 "Scaling deck needed: set up early, play Withers out",
    "ceremonialbeast": "Plow: stunned the first time it drops to 150 HP. Push it there to skip a big hit. "
                       "Beast Cry (Ringing): you can play only ONE card that turn, make it count",
    "crusher": "Kaiser Crab: Crusher and Rocket. Crab Rage: when one dies, the other gains +5 Strength and 99 Block, "
               "so bring both low and finish them close together. You take 50% more from the one behind you; targeting one turns you to face it",
    "rocket": "Kaiser Crab: Rocket's Charge Up -> Laser is the big hit; it sleeps (Recharge) after. "
              "Crab Rage: when one dies the other gets +5 Strength and 99 Block",
    "knowledgedemon": "Curse of Knowledge makes you pick a curse; Ponder heals it 30 and adds Strength. "
                      "Burst it between heals",
    "lagavulinmatriarch": "Asleep 3 turns: play Powers and set up. It gains Plating 12; Soul Siphon takes 2 "
                          "Strength and 2 Dexterity from you",
    "queen": "Turn 1 binds your first 3 draws each turn; turn 2 You're Mine gives you 99 Frail, Weak and "
             "Vulnerable. While Torch Head Amalgam lives she only blocks and gives it Strength, so it keeps "
             "growing; once it dies she attacks (5 hits, then Execution). Finish your setup, then kill the Amalgam",
    "torchheadamalgam": "The Queen's minion: it gains Strength every turn while alive; killing it starts her attacks",
    "soulfysh": "Fade: Intangible 2 (every hit does 1) -- Block and set up those turns, don't attack. "
                "Beckon cards cost HP if still in hand at end of turn",
    "testsubject": "Three lives (Adaptable: revives stronger). Enrage: +Strength each Skill you play; Nemesis: "
                   "Intangible every other turn; Painful Stabs adds Wounds. Save burst for non-Intangible turns",
    "theinsatiable": "Sandpit counts down each turn and KILLS you at 0. Liquify adds 6 Frantic Escape: each one "
                     "you play pushes the pit back (costs rise). Play enough of them, kill it fast",
    "kinpriest": "The Kin Followers are minions: they leave when the Priest dies. Usually burst the Priest; "
                 "Ritual adds Strength each time",
    "kinfollower": "Minion: leaves when the Kin Priest dies. Power Dance adds Strength",
    "vantom": "Slippery: its next HP loss is only 1, so open with your weakest hit. "
              "Dismember adds 3 Wounds; Prepare adds Strength",
    "waterfallgiant": "Steam Eruption: when it dies it explodes at the end of your NEXT turn, so keep Block for "
                      "after the kill. Siphon heals it; the pressure gun grows",
}

#: encounter id (map) -> monster names in it.
ENCOUNTERS: dict[str, list[str]] = {
    "bygoneeffigy": ["bygoneeffigy"],
    "byrdonis": ["byrdonis"],
    "decimillipede": ["decimillipedesegment"],
    "entomancer": ["entomancer"],
    "infestedprisms": ["infestedprism"],
    "knights": ["spectralknight", "flailknight", "magiknight"],
    "mechaknight": ["mechaknight"],
    "phantasmalgardeners": ["phantasmalgardener"],
    "phrogparasite": ["phrogparasite"],
    "skulkingcolony": ["skulkingcolony"],
    "soulnexus": ["soulnexus"],
    "terroreel": ["terroreel"],
    "aeonglass": ["aeonglass"],
    "ceremonialbeast": ["ceremonialbeast"],
    "kaisercrab": ["crusher", "rocket"],
    "knowledgedemon": ["knowledgedemon"],
    "lagavulinmatriarch": ["lagavulinmatriarch"],
    "queen": ["queen"],
    "soulfysh": ["soulfysh"],
    "testsubject": ["testsubject"],
    "theinsatiable": ["theinsatiable"],
    "thekin": ["kinpriest"],
    "vantom": ["vantom"],
    "waterfallgiant": ["waterfallgiant"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _monster_key(name: str) -> str | None:
    k = _norm(name)
    if len(k) < 4:
        return None
    if k in TIPS:
        return k
    for key in TIPS:
        if k.startswith(key) or key.startswith(k):
            return key
    return None


def combat_lines(state: dict[str, Any]) -> list[str]:
    """Tips for the elites/bosses in this fight (one per kind)."""
    seen: list[str] = []
    for e in state.get("enemies") or []:
        if not isinstance(e, dict):
            continue
        key = _monster_key(str(e.get("id") or "")) or _monster_key(str(e.get("name") or ""))
        if key and key not in seen:
            seen.append(key)
    return [f"How to fight it: {TIPS[k]}" for k in seen]


def encounter_tip(encounter_id: str) -> str | None:
    """Tip for a map encounter id like ``TERROR_EEL_ELITE``."""
    k = _norm(re.sub(r"_(ELITE|BOSS)$", "", encounter_id))
    monsters = ENCOUNTERS.get(k)
    if not monsters:
        return None
    return " ".join(TIPS[m] + "." for m in monsters[:1] if m in TIPS)
