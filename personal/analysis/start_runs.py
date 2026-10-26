"""Full-run test of starting packages: each package's cards are added to the
starter deck (RunManager.force_add_card), then the character's shipped AI
(champions.toml ONNX via sts2_rs.PpoRunAgent: PPO net for run decisions,
eval-cheap MCTS combat) plays whole A10 runs. Same seeds for every package
of a character, plus a no-extra-cards baseline. Idle priority, resumable.

usage (from sts2-pycli, PYTHONPATH=python): python start_runs.py OUT.jsonl SEEDS [PROCS]"""
import ctypes
import json
import multiprocessing as mp
import os
import sys

PACKAGES = {
    "Ironclad": {
        "baseline": [],
        "sim-best": ["DOMINATE", "FIGHT_ME", "OUTRAGE", "INFERNO"],
        "pro-favored": ["OFFERING", "COLOSSUS", "CRUELTY", "BATTLE_TRANCE"],
        "mix A": ["DOMINATE", "FIGHT_ME", "COLOSSUS", "BATTLE_TRANCE"],
        "mix B": ["OFFERING", "FIGHT_ME", "OUTRAGE", "COLOSSUS"],
    },
    "Silent": {
        "baseline": [],
        "sim-best": ["BLADE_OF_INK", "BOUNCING_FLASK", "NOXIOUS_FUMES", "HAZE"],
        "pro-favored": ["ADRENALINE", "FOOTWORK", "ACROBATICS", "EXPOSE"],
        "mix A": ["AFTERIMAGE", "BOUNCING_FLASK", "NOXIOUS_FUMES", "FOOTWORK"],
        "mix B": ["ADRENALINE", "BOUNCING_FLASK", "NOXIOUS_FUMES", "FOOTWORK"],
    },
    "Defect": {
        "baseline": [],
        "sim-best": ["ECHO_FORM", "DARKNESS", "SHADOW_SHIELD", "NULL"],
        "pro-favored": ["ECHO_FORM", "GLACIER", "CAPACITOR", "OVERCLOCK"],
        "mix A": ["DEFRAGMENT", "DARKNESS", "GLACIER", "BALL_LIGHTNING"],
        "mix B": ["ECHO_FORM", "DARKNESS", "GLACIER", "LIGHTNING_ROD"],
    },
    "Necrobinder": {
        "baseline": [],
        "sim-best": ["CALL_OF_THE_VOID", "BONE_SHARDS", "REAP", "DEATHBRINGER"],
        "pro-favored": ["DEMESNE", "ENFEEBLING_TOUCH", "DEBILITATE", "LETHALITY"],
        "mix A": ["CALL_OF_THE_VOID", "ENFEEBLING_TOUCH", "BONE_SHARDS", "DEBILITATE"],
        "mix B": ["DEMESNE", "BONE_SHARDS", "REAP", "ENFEEBLING_TOUCH"],
    },
    "Regent": {
        "baseline": [],
        "sim-best": ["BIG_BANG", "BULWARK", "FURNACE", "WROUGHT_IN_WAR"],
        "pro-favored": ["BIG_BANG", "BULWARK", "REFLECT", "ORBIT"],
        "mix A": ["VOID_FORM", "BULWARK", "REFLECT", "CHARGE"],
        "mix B": ["BIG_BANG", "BULWARK", "CONVERGENCE", "CHARGE"],
    },
}


def idle():
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)


def run(job):
    import sts2_client
    import sts2_rs
    import champions

    char, pkg, cards, seed = job
    try:
        rm = sts2_client.rs_engine_call(sts2_rs.RunManager, version="v0.111.0", seed=seed, character=char,
                                        ascension_level=10, neow_start=True, act1="roll")
        for c in cards:
            rm.force_add_card(c)
        agent = sts2_rs.PpoRunAgent(seed=seed, sims=1000, boss_sims=4000,
                                    model_path=champions.onnx_path(char))
        steps = 0
        while not (rm.is_over() if callable(rm.is_over) else rm.is_over) and steps < 20000:
            if agent.step(rm) is None:
                break
            steps += 1
        def val(x):
            return x() if callable(x) else x

        return {"char": char, "pkg": pkg, "seed": seed, "won": bool(val(rm.player_won)),
                "floor": int(val(rm.total_floor)), "over": bool(val(rm.is_over)), "steps": steps}
    except Exception as e:
        return {"char": char, "pkg": pkg, "seed": seed, "error": repr(e)[:200]}


if __name__ == "__main__":
    out, n = sys.argv[1], int(sys.argv[2])
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    idle()
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if "error" not in r:
                done.add((r["char"], r["pkg"], r["seed"]))
    jobs = [(c, p, cards, s) for c, pk in PACKAGES.items() for p, cards in pk.items()
            for s in range(1, n + 1) if (c, p, s) not in done]
    print(f"{len(jobs)} runs to go", flush=True)
    with mp.Pool(procs, initializer=idle) as pool, open(out, "a") as f:
        for i, row in enumerate(pool.imap_unordered(run, jobs, chunksize=1)):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, row, flush=True)
