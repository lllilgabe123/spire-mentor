"""Does the setup-first reorder (realgame_turnplan.reorder_setup) keep or
improve live picks? On logged combat turns: live search (16,000 sims,
known draws) -> turn plan -> reorder. When the first card changes, judge
old vs new first move with an independent forced search (4 x 6,000).

usage (from sts2-pycli, PYTHONPATH=python): python setup_check.py N"""
import glob
import json
import os
import random
import sys

import sts2_rs
from sts2_hint import adapter

import realgame_turnplan as tp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
table = tp.setup_table()
L = os.path.join(os.environ["LOCALAPPDATA"], "SpireMentor", "logs")
states = []
for f in glob.glob(os.path.join(L, "*.jsonl")):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        d = r["data"]
        if r["type"] == "state" and d.get("decision") == "combat_play" and \
                any(str(c.get("id")) in table for c in d.get("hand") or []):
            states.append(d)
random.seed(int(os.environ.get("SAMPLE_SEED", "7")))
random.shuffle(states)


def truth(cs, act):
    rows = [sts2_rs.mcts_score_action_forced(cs, act, sims=6000, seed=s, hidden_info="beliefdraw") for s in (301, 302, 303, 304)]
    rows = [r for r in rows if r]
    return sum(r["p_win"] for r in rows) / len(rows), sum(r["mean_final_hp"] for r in rows) / len(rows)


done = reordered = first_changed = better = worse = 0
for st in states:
    if done >= N:
        break
    try:
        wire = adapter.translate(st)
        wire["deck"] = dict(wire["deck"])
        wire["deck"]["pinned_top"] = list(range(len(wire["deck"].get("draw_pile") or [])))
        cs = sts2_rs.state_from_dict(wire)
        res = sts2_rs.mcts_recommend(cs, sims=16000, seed=1, hidden_info="beliefdraw", threads=8)
    except Exception:
        continue
    hand = wire["deck"]["hand"]
    b = next(a["action"] for a in res["actions"] if a["is_best"])
    if b["action"] != "play":
        continue
    done += 1
    best = {"action": "play", "card_index": hand[b["card_index"]]["stable_idx"], "target_index": b.get("target_index")}
    steps = []
    try:
        order, line = tp.plan_turn(sts2_rs, cs, best, hand, {}, sims=1600, steps_out=steps)
        redo = tp.reorder_setup(sts2_rs, cs, steps, table)
    except Exception as e:
        print("skip:", e)
        continue

    if not redo:
        continue
    reordered += 1
    first = redo[2]
    if first.get("handle") == best["card_index"]:
        continue
    first_changed += 1
    i = next(k for k, c in enumerate(hand) if c.get("stable_idx") == first["handle"])
    new_act = {"action": "play", "card_index": i, "target_index": first.get("target")}
    (pa, ha), (pb, hb) = truth(cs, b), truth(cs, new_act)
    better += pb > pa + 0.005 or (abs(pb - pa) <= 0.005 and hb > ha + 0.5)
    worse += pb < pa - 0.005 or (abs(pb - pa) <= 0.005 and hb < ha - 0.5)
    print(f"{line[:4]} -> {redo[1][:4]} | old {pa:.1%} {ha:.1f}hp  new {pb:.1%} {hb:.1f}hp", flush=True)
print(f"\n{done} turns, {reordered} plans reordered, {first_changed} changed the first card: "
      f"{better} better, {worse} worse, {first_changed - better - worse} same")
