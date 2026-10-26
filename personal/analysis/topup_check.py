"""Does realgame_topup.double_check improve live picks? Replays combat turns
from the player's run logs through the live search (16,000 sims, known
draws), applies the double-check, and judges each switch with a bigger,
independent forced search (4 seeds x 8,000 per move, other seeds).

usage (from sts2-pycli, PYTHONPATH=python): python topup_check.py N_STATES"""
import glob
import json
import os
import random
import sys
import time

import sts2_rs
from sts2_hint import adapter

import realgame_topup as tu

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
L = os.path.join(os.environ["LOCALAPPDATA"], "SpireMentor", "logs")
states = []
for f in glob.glob(os.path.join(L, "*.jsonl")):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r["type"] == "state" and r["data"].get("decision") == "combat_play" and r["data"].get("hand"):
            states.append(r["data"])
random.seed(int(os.environ.get("SAMPLE_SEED", "7")))
random.shuffle(states)


def truth(cs, action):
    rows = [sts2_rs.mcts_score_action_forced(cs, action, sims=8000, seed=s, hidden_info="beliefdraw") for s in (201, 202, 203, 204)]
    return sum(r["p_win"] for r in rows) / 4


done = switched = better = worse = 0
gain = 0.0
t_main = t_top = 0.0
for st in states:
    if done >= N:
        break
    try:
        wire = adapter.translate(st)
        wire["deck"] = dict(wire["deck"])
        wire["deck"]["pinned_top"] = list(range(len(wire["deck"].get("draw_pile") or [])))
        cs = sts2_rs.state_from_dict(wire)
        t0 = time.time()
        res = sts2_rs.mcts_recommend(cs, sims=16000, seed=1, hidden_info="beliefdraw", threads=8)
        t_main += time.time() - t0
    except Exception:
        continue
    hand = wire["deck"]["hand"]
    total = sum(a.get("visits") or 0 for a in res["actions"]) or 1
    options = []
    for a in res["actions"]:
        act = a["action"]
        if act.get("action") == "end_turn":
            o = {"label": "End turn"}
        elif act.get("action") == "play":
            o = {"label": f"card{act['card_index']}->{act.get('target_index')}", "card_index": hand[act["card_index"]].get("stable_idx"),
                 "target_index": act.get("target_index")}
        else:
            o = {"label": str(act)}
        o.update(q=a.get("q"), visit_share=(a.get("visits") or 0) / total, is_best=bool(a.get("is_best")), _act=act)
        options.append(o)
    if len(options) < 2:
        continue
    done += 1
    advice = {"options": options, "best": {}}
    old = next(o for o in options if o["is_best"])
    t0 = time.time()
    note = tu.double_check(sts2_rs, cs, advice, hand, sims=4000, hidden_info="beliefdraw")
    t_top += time.time() - t0
    if note:
        new = next(o for o in options if o["is_best"])
        a, b = truth(cs, old["_act"]), truth(cs, new["_act"])
        switched += 1
        better += b > a + 0.005
        worse += b < a - 0.005
        gain += b - a
        print(f"switch {old['label']} ({a:.1%}) -> {new['label']} ({b:.1%})", flush=True)
print(f"\n{done} turns checked, {switched} switched: {better} better, {worse} worse, "
      f"avg win-chance change {gain / max(switched, 1):+.2%} per switch")
print(f"time per turn: main search {t_main / max(done, 1):.2f}s, double-check {t_top / max(done, 1):.2f}s")
