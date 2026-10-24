"""Replay SimTieBreak's rule over every logged fight decision.

Mirrors SimTieBreak.cs: options within TIE_Q of the best Q and TIE_WIN of its
win chance are "about equal"; among them a Power card goes first, then a
self-targeted non-attack card that buffs Strength/Dexterity/Vigor/Focus (here
read from the card text; the mod checks the card's PowerVar), then the
search's own pick. Reports how often it would label a tie and how often it
would change the recommendation, with examples. Read-only.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

TIE_Q, TIE_WIN = 0.005, 0.01
# "End turn" while a card is still playable: a wider margin (~3 HP), and
# End turn ranks last among the tied moves.
END_TURN_Q = 0.01
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "SpireMentor" / "logs"
BUFF = re.compile(r"\b(Strength|Dexterity|Vigor|Focus)\b")


def name(label: str) -> str:
    return label.split(" → ")[0].strip()


DEBUFF = re.compile(r"\b(Vulnerable|Weak)\b")


def rank(opt: dict, hand: list[dict]) -> int:
    """0 power, 1 self-buff, 2 Vulnerable/Weak on enemies, 3 other, 4 End turn."""
    ci = opt.get("card_index")
    if opt.get("label") == "End turn":
        return 4
    if ci is None or not (0 <= ci < len(hand)):
        return 3
    card = hand[ci]
    target = str(card.get("target_type", "")).lower()
    text = str(card.get("description", ""))
    if card.get("type") == "Power":
        return 0
    if card.get("type") != "Attack" and target == "self" and BUFF.search(text):
        return 1
    if "enemy" in target or "enemies" in target or "→" in opt.get("label", ""):
        if DEBUFF.search(text):
            return 2
    return 3


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # labels contain "→"
    only = sys.argv[1].lower() if len(sys.argv) > 1 else None  # e.g. "inflame"
    decisions = ties = changed = 0
    changes: Counter[str] = Counter()
    examples: list[str] = []
    for path in sorted(LOG_DIR.glob("2026*.jsonl")):
        hand: list[dict] = []
        for line in path.open(encoding="utf-8", errors="replace"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = rec.get("data") or {}
            if rec.get("type") == "state":
                hand = data.get("hand") or hand
                continue
            if rec.get("type") != "mcts":
                continue
            adv = data.get("advice") or {}
            opts = [o for o in adv.get("options", []) if o.get("q") is not None]
            best = next((o for o in opts if o.get("is_best")), None)
            if adv.get("p_win") is None or best is None:
                continue
            decisions += 1
            margin = END_TURN_Q if best.get("label") == "End turn" else TIE_Q
            tied = [o for o in opts if best["q"] - o["q"] <= margin
                    and abs((o.get("p_win") or 0) - (best.get("p_win") or 0)) <= TIE_WIN]
            if best.get("label") == "End turn":
                # Against End turn only cards that cannot hurt you: attacks,
                # powers, self-buffs (not e.g. Bloodletting, which costs HP).
                def harmless(o):
                    ci = o.get("card_index")
                    if ci is None or not (0 <= ci < len(hand)):
                        return False
                    if re.search(r"\blose\b[^.]*\bHP\b", str(hand[ci].get("description", "")), re.I):
                        return False  # costs HP (the mod checks HpLossVar)
                    return hand[ci].get("type") == "Attack" or rank(o, hand) <= 2
                tied = [best] + [o for o in tied if o is not best and harmless(o)]
            if len({name(o["label"]) for o in tied}) < 2:
                continue
            ties += 1
            chosen = sorted(tied, key=lambda o: (rank(o, hand), 0 if o.get("is_best") else 1, -o["q"]))[0]
            if chosen is not best:
                changed += 1
                key = f"{name(best['label'])} -> {name(chosen['label'])}"
                changes[key] += 1
                if (only is None and len(examples) < 6) or (only and only in key.lower()):
                    examples.append(f"  {path.name[:15]}  AI said {best['label']} (q {best['q']:.4f}), "
                                    f"tie-break says {chosen['label']} (q {chosen['q']:.4f})")
    print(f"fight decisions: {decisions}")
    print(f"labelled 'about equal' (2+ different cards within {TIE_Q} Q): {ties} ({100 * ties / max(decisions, 1):.0f}%)")
    print(f"recommendation changed by the tie-break: {changed} ({100 * changed / max(decisions, 1):.0f}%)")
    print("most common changes:")
    for k, n in changes.most_common(8):
        print(f"  {n:4}x  {k}")
    print("examples:")
    print("\n".join(examples))


if __name__ == "__main__":
    main()
