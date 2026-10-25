"""Leaf value correction for the combat search (GH #830).

The search scores a leaf by ONE rl2 playout. From the strong human's blocking
states that playout finishes fights slowly and leaks HP the searching pilot
does not (`docs/experiments/2026-10-23-830-hp-weight.md`, step 5), so the
leaf under-prices blocking lines. This module learns a residual

    r(s) ~ E[ V_search(s) - V_playout(s) | features(s) ]

where `V_search` is what the searching pilot realizes from `s` (the value the
search backs up for the fight's final state, `sts2_rs.search_playout_value`)
and `V_playout` the mean rl2 playout return from `s`
(`sts2_rs.values_many`, belief view). The leaf becomes `playout + lambda *
r(s)`. `r` reads the rl2 playout net's v1 state features
(`sts2_rs.rollout_features(state)["state"]`, 32 floats), so the Rust leaf can
compute it with the code the playout already runs.

    gen    play replayed human positions with the deployed pilot to the end
           of the fight; at each decision record the state features, the
           mean of K belief playouts from it, and (at the end) the realized
           value. Starts as for `rollout_learn gen-replay`.
    train  fit a small MLP `r` (L2 + shrinkage toward 0), write JSON weights.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time
import zlib
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from ppo.combat_bench import agents

STATE_DIM = 32


def play_and_label(
    rm: Any,
    agent: Any,
    k: int,
    seed: int,
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
    keep_state: bool = False,
) -> dict[str, Any]:
    """Let `agent` finish `rm`'s combat. Before each searched player decision
    append `{"x", "pm", ...meta}` to `rows` (`x` = v1 state features, `pm` =
    mean belief-playout return of `k` playouts); returns `{"vt", "won",
    "steps"}`, `vt` = the search's value of the final state, rebuilt by
    replaying the last action on a clone (the RunManager drops the combat
    state once the fight ends). `vt` is None when that fails."""
    import sts2_rs

    first = len(rows)
    steps = 0
    pre = None
    last = None
    while rm.phase() == "combat" and not rm.is_over() and steps < 3000:
        cs = rm.combat_state()
        if cs is None:
            break
        g = cs.get_state()
        if k and g.get("type") == "combat_action" and not cs.terminal_value()[0]:
            vals = sts2_rs.values_many(
                [cs] * k, [seed * 1000 + steps * 7 + i for i in range(k)], mode="belief"
            )
            rows.append(
                {
                    **meta,
                    "x": sts2_rs.rollout_features(cs, version=1)["state"],
                    "pm": float(np.mean([v["search_value"] for v in vals])),
                    "pwin": float(np.mean([bool(v["won"]) for v in vals])),
                    "round": int(g.get("round") or 0),
                    "hp": int((g.get("player") or {}).get("hp") or 0),
                    # The decision's state (wire dict), for features computed
                    # later (`state_from_dict`).
                    **({"s": sts2_rs.state_to_dict(cs)} if keep_state else {}),
                }
            )
        pre = cs.clone()
        rec = agent.step(rm)
        steps += 1
        if rec is None:
            break
        search = rec.get("combat_search") or {}
        last = search.get("recommended") or rec.get("action")
    ended = rm.phase() != "combat" or rm.is_over()
    # A death ends the run (`is_over`); a win moves the run past combat.
    won = ended and not (rm.is_over() and not rm.player_won)
    out: dict[str, Any] = {"vt": None, "won": won, "steps": steps}
    if ended and pre is not None and last is not None:
        end = pre.clone()
        try:
            end.apply_step(last)
            done, tv = end.terminal_value()
            if done and (tv > 0) == won:
                out["vt"] = float(sts2_rs.search_playout_value(end))
        except ValueError:
            pass
    for r in rows[first:]:
        r["vt"] = out["vt"]
    return out


def gen_fight(job: tuple[Any, ...]) -> dict[str, Any]:
    path, combat_id, starts, sims, k, keep_state = job
    import sts2_rs

    from ppo.bc_human import replay
    from ppo.combat_bench import takeover
    from ppo.turnplan.pro import fight_turns, room_type

    agents.apply_env({})
    name = Path(path).name
    out: dict[str, Any] = {"keys": [], "rows": [], "skipped": 0, "no_vt": 0}
    try:
        rec = next(r for r in replay.fights(path) if r["combat_id"] == combat_id)
        turns, _ = fight_turns(rec)
        room = room_type(rec["encounter"])
        for t, mode in starts:
            key = f"{name}:{combat_id}:{t}:{mode}"
            out["keys"].append(key)
            if t >= len(turns) or turns[t]["status"] != "legal":
                out["skipped"] += 1
                continue
            state = turns[t]["state"]
            seed = zlib.crc32(key.encode()) % (2**31)
            rm = sts2_rs.RunManager(
                seed=seed,
                character=rec["character"].capitalize(),
                ascension_level=rec["ascension"] or 0,
            )
            rm.debug_enter_combat(state.clone(), room)
            if mode == "human":
                check = state.clone()
                ok = True
                for a in turns[t]["actions"]:
                    ra = takeover.run_action(a)
                    if ra is None:
                        ok = False
                        break
                    rm.take_action(ra)
                    check.apply_step(a)
                    if rm.phase() != "combat" or rm.is_over():
                        break
                cs = rm.combat_state() if rm.phase() == "combat" and not rm.is_over() else None
                if not ok or cs is None or takeover._snapshot(cs) != takeover._snapshot(check):
                    out["skipped"] += 1
                    continue
            agent = sts2_rs.MctsCombatAgent(seed=seed, sims=sims, boss_sims=sims)
            meta = {"run": name, "start": key, "room": room}
            res = play_and_label(rm, agent, k, seed, out["rows"], meta, keep_state)
            out["no_vt"] += res["vt"] is None
    except Exception as e:
        out["error"] = repr(e)[:300]
    return out


def gen(args: argparse.Namespace) -> int:
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    sel = json.loads(Path(args.collect, "selection.json").read_text(encoding="utf-8"))["files"]
    path_of = {Path(r["file"]).name: r["file"] for r in sel}
    by_fight: dict[tuple[str, str], list[tuple[int, str]]] = collections.defaultdict(list)
    with open(args.starts, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_fight[r["file"], r["combat_id"]].append((int(r["turn"]), r["mode"]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.txt"
    done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
    jobs = []
    for (f, c), st in sorted(by_fight.items()):
        st = [s for s in st if f"{f}:{c}:{s[0]}:{s[1]}" not in done]
        if st:
            jobs.append((path_of[f], c, st, args.sims, args.k, args.keep_states))
    random.Random(args.sample_seed).shuffle(jobs)
    print(
        f"[leaf_correction gen] {len(jobs)} fights, {sum(len(j[2]) for j in jobs)} starts",
        flush=True,
    )
    t0 = time.time()
    n = n_rows = err = skipped = no_vt = 0
    with (
        get_context("spawn").Pool(args.procs) as pool,
        open(out_dir / "rows.jsonl", "a", encoding="utf-8") as rows_f,
        open(done_path, "a", encoding="utf-8") as done_f,
    ):
        for res in pool.imap_unordered(gen_fight, jobs):
            n += 1
            err += "error" in res
            skipped += res["skipped"]
            no_vt += res["no_vt"]
            for r in res["rows"]:
                rows_f.write(json.dumps(r) + "\n")
            n_rows += len(res["rows"])
            for k in res["keys"]:
                done_f.write(k + "\n")
            rows_f.flush()
            done_f.flush()
            if n % 50 == 0 or n == len(jobs):
                print(
                    f"[leaf_correction gen] {n}/{len(jobs)} fights, {n_rows} rows, {err} errors, "
                    f"{skipped} skipped starts, {no_vt} without a final value, {time.time() - t0:.0f}s",
                    flush=True,
                )
    print("DONE", flush=True)
    return 0


# ── the residual model ──────────────────────────────────────────────────


def forward(net: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """`r(x)` for feature rows `x [n, 32]` (numpy twin of the Rust leaf):
    standardize, one ReLU layer, linear out, scale and clip."""
    z = (np.asarray(x, np.float64) - np.asarray(net["mean"])) / np.asarray(net["std"])
    h = np.maximum(z @ np.asarray(net["w1"]).T + np.asarray(net["b1"]), 0.0)
    r = h @ np.asarray(net["w2"]) + float(net["b2"])
    return np.clip(r, -net["clip"], net["clip"])


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    """Every labelled row; `_i` is its line number across `paths` (the row
    index of a `--extra` feature array written for the same files)."""
    rows = []
    i = 0
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                r["_i"] = i
                i += 1
                if r.get("vt") is not None:
                    rows.append(r)
    return rows


INPUT_BLOCKS = ("x", "K", "C")


def input_matrix(
    rows: list[dict[str, Any]], inputs: list[str], extra: dict[str, np.ndarray]
) -> np.ndarray:
    """The model input: the blocks `inputs` in order, `x` = the row's 32 rl2
    state features, any other block = `extra[block]` at the row's `_i`."""
    idx = np.asarray([r["_i"] for r in rows])
    parts = []
    for b in inputs:
        if b == "x":
            parts.append(np.asarray([r["x"] for r in rows], np.float32))
        else:
            parts.append(np.asarray(extra[b], np.float32)[idx])
    return np.concatenate(parts, 1)


def train(args: argparse.Namespace) -> int:
    import torch

    rows = load_rows(args.rows)
    if args.turn_starts:
        # Keep each start's first decision of every round: turn-start states,
        # the kind of state a leaf below an EndTurn chance node starts from.
        seen: set[tuple[str, int]] = set()
        kept = []
        for r in rows:
            k = (r["start"], r["round"])
            if k not in seen:
                seen.add(k)
                kept.append(r)
        rows = kept
    runs = sorted({r["run"] for r in rows})
    rng = random.Random(args.seed)
    val_runs = set(rng.sample(runs, max(1, int(len(runs) * args.val_frac))))
    inputs = args.inputs.split(",")
    extra = dict(np.load(args.extra)) if args.extra else {}
    for b in inputs:
        # A block with a validity mask `<b>_ok` drops the rows it could not
        # compute (after the run split, so every variant splits alike).
        if f"{b}_ok" in extra:
            rows = [r for r in rows if extra[f"{b}_ok"][r["_i"]]]
    X = input_matrix(rows, inputs, extra)
    y = np.asarray([r["vt"] - r["pm"] for r in rows], np.float32)
    is_val = np.asarray([r["run"] in val_runs for r in rows])
    # Weight each start's decisions to sum to 1, so a long fight (many
    # decisions, one realized outcome) does not dominate.
    per_start = collections.Counter(r["start"] for r in rows)
    w = np.asarray([1.0 / per_start[r["start"]] for r in rows], np.float32)
    mean, std = X[~is_val].mean(0), X[~is_val].std(0) + 1e-3
    dev = torch.device(args.device)
    t = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    Xt, yt, wt = t((X - mean) / std), t(y), t(w)
    tr, va = t(np.flatnonzero(~is_val)), t(np.flatnonzero(is_val))
    torch.manual_seed(args.seed)
    l1 = torch.nn.Linear(X.shape[1], args.hidden).to(dev)
    l2 = torch.nn.Linear(args.hidden, 1).to(dev)
    torch.nn.init.zeros_(l2.weight)
    torch.nn.init.zeros_(l2.bias)
    params = [*l1.parameters(), *l2.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)

    def pred(idx: Any) -> Any:
        return l2(torch.relu(l1(Xt[idx])))[:, 0]

    def loss_of(idx: Any) -> float:
        with torch.no_grad():
            p = pred(idx)
            return float(((p - yt[idx]) ** 2 * wt[idx]).sum() / wt[idx].sum())

    base_va = float(
        ((yt[va] - (yt[tr] * wt[tr]).sum() / wt[tr].sum()) ** 2 * wt[va]).sum() / wt[va].sum()
    )
    print(
        f"[leaf_correction train] {len(rows)} rows, {len(runs)} runs ({len(val_runs)} val), "
        f"label mean {y.mean():+.4f} sd {y.std():.4f}, constant-fit val mse {base_va:.5f}",
        flush=True,
    )
    for epoch in range(args.epochs):
        perm = tr[torch.randperm(len(tr), device=dev)]
        for b in torch.split(perm, args.batch):
            p = pred(b)
            mse = ((p - yt[b]) ** 2 * wt[b]).sum() / wt[b].sum()
            # Shrink the output toward 0 (not just the weights): r must stay
            # a small correction to the playout, never a leaf of its own.
            reg = args.shrink * (p**2 * wt[b]).sum() / wt[b].sum()
            reg = reg + args.wd * sum((q**2).sum() for q in (l1.weight, l2.weight))
            opt.zero_grad()
            (mse + reg).backward()
            opt.step()
        print(
            f"[leaf_correction train] epoch {epoch + 1}: train mse {loss_of(tr):.5f} val mse {loss_of(va):.5f}",
            flush=True,
        )
    w_ = lambda a: a.detach().cpu().numpy().astype(float).round(7).tolist()  # noqa: E731
    net = {
        "version": 1,
        "feature_version": 1,
        "state_dim": int(X.shape[1]),
        "inputs": inputs,
        "hidden": args.hidden,
        "mean": mean.astype(float).round(7).tolist(),
        "std": std.astype(float).round(7).tolist(),
        "w1": w_(l1.weight),
        "b1": w_(l1.bias),
        "w2": w_(l2.weight[0]),
        "b2": float(l2.bias.item()),
        "clip": args.clip,
        "meta": {
            "rows": args.rows,
            "n": len(rows),
            "val_runs": sorted(val_runs),
            "val_mse": loss_of(va),
            "constant_val_mse": base_va,
            "shrink": args.shrink,
            "wd": args.wd,
        },
    }
    Path(args.out).write_text(json.dumps(net), encoding="utf-8")
    print(f"[leaf_correction train] wrote {args.out}", flush=True)
    return 0


# ── paired training (contrast pairs) ────────────────────────────────────

ALT_KEY = {"x": "x", "K": "k", "C": "c"}


def load_pairs(paths: list[str], inputs: list[str]) -> list[dict[str, Any]]:
    """Positions with >= 2 usable alternatives: each alternative's input
    vector (`inputs` blocks), mean playout `pm` and mean realized `vt`."""
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                alts = []
                for a in r.get("alts") or []:
                    vt = [v for v in a.get("vt") or [] if v is not None]
                    if not vt or any(a.get(ALT_KEY[b]) is None for b in inputs):
                        continue
                    alts.append(
                        {
                            "kind": a["kind"],
                            "z": np.concatenate(
                                [np.asarray(a[ALT_KEY[b]], np.float32) for b in inputs]
                            ),
                            "pm": float(np.mean(a["pm"])),
                            "vt": float(np.mean(vt)),
                        }
                    )
                if len(alts) >= 2:
                    out.append(
                        {
                            "run": r["file"],
                            "key": f"{r['file']}:{r['combat_id']}:{r['turn']}",
                            "alts": alts,
                        }
                    )
    return out


def train_pairs(args: argparse.Namespace) -> int:
    """Fit r so that r(a) - r(b) ~ (vt_a - vt_b) - (pm_a - pm_b) for every
    pair of alternatives of one position (each position weighted 1), plus
    `--anchor` x r^2 keeping the level near 0; the clip and the output form
    are `train`'s."""
    import torch

    inputs = args.inputs.split(",")
    pos = load_pairs(args.pairs, inputs)
    # The run split comes from every position with rl2 features, so input
    # sets that drop some alternatives (C) still split alike.
    runs = sorted({p["run"] for p in load_pairs(args.pairs, ["x"])})
    rng = random.Random(args.seed)
    val_runs = set(rng.sample(runs, max(1, int(len(runs) * args.val_frac))))
    Z, PM, VT, POS, spans = [], [], [], [], []
    for i, p in enumerate(pos):
        first = len(Z)
        for a in p["alts"]:
            Z.append(a["z"])
            PM.append(a["pm"])
            VT.append(a["vt"])
            POS.append(i)
        spans.append(range(first, len(Z)))
    Z = np.asarray(Z, np.float32)
    POS = np.asarray(POS)
    I, J, W = [], [], []
    for idx in spans:
        pairs = [(a, b) for k, a in enumerate(idx) for b in idx[k + 1 :]]
        for a, b in pairs:
            I.append(a)
            J.append(b)
            W.append(1.0 / len(pairs))
    I, J, W = np.asarray(I), np.asarray(J), np.asarray(W, np.float32)
    PM, VT = np.asarray(PM, np.float32), np.asarray(VT, np.float32)
    D = (VT[I] - VT[J]) - (PM[I] - PM[J])
    is_val_alt = np.asarray([pos[k]["run"] in val_runs for k in POS])
    is_val = is_val_alt[I]
    mean, std = Z[~is_val_alt].mean(0), Z[~is_val_alt].std(0) + 1e-3
    dev = torch.device(args.device)
    t = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    Zt, Dt, Wt, It, Jt = t((Z - mean) / std), t(D), t(W), t(I), t(J)
    tr, va = t(np.flatnonzero(~is_val)), t(np.flatnonzero(is_val))
    torch.manual_seed(args.seed)
    l1 = torch.nn.Linear(Z.shape[1], args.hidden).to(dev)
    l2 = torch.nn.Linear(args.hidden, 1).to(dev)
    torch.nn.init.zeros_(l2.weight)
    torch.nn.init.zeros_(l2.bias)
    opt = torch.optim.Adam([*l1.parameters(), *l2.parameters()], lr=args.lr)

    def r_of(rows: Any) -> Any:
        return torch.clamp(l2(torch.relu(l1(Zt[rows])))[:, 0], -args.clip, args.clip)

    def loss_of(pidx: Any) -> tuple[Any, Any]:
        ri, rj = r_of(It[pidx]), r_of(Jt[pidx])
        mse = ((ri - rj - Dt[pidx]) ** 2 * Wt[pidx]).sum() / Wt[pidx].sum()
        anchor = ((ri**2 + rj**2) * Wt[pidx]).sum() / (2 * Wt[pidx].sum())
        return mse, anchor

    with torch.no_grad():
        base = float((Dt[va] ** 2 * Wt[va]).sum() / Wt[va].sum())
    print(
        f"[leaf_correction train-pairs] {len(pos)} positions, {len(Z)} alternatives, {len(I)} pairs, "
        f"{len(runs)} runs ({len(val_runs)} val); zero-r val mse {base:.5f}",
        flush=True,
    )
    import copy

    best = (float("inf"), None, 0)
    for epoch in range(args.epochs):
        perm = tr[torch.randperm(len(tr), device=dev)]
        for b in torch.split(perm, args.batch):
            mse, anchor = loss_of(b)
            reg = args.anchor * anchor + args.wd * ((l1.weight**2).sum() + (l2.weight**2).sum())
            opt.zero_grad()
            (mse + reg).backward()
            opt.step()
        with torch.no_grad():
            v_mse, _ = loss_of(va)
            t_mse, _ = loss_of(tr)
        # Early stopping on the held-out runs: keep the best epoch's weights.
        if float(v_mse) < best[0]:
            best = (
                float(v_mse),
                (copy.deepcopy(l1.state_dict()), copy.deepcopy(l2.state_dict())),
                epoch + 1,
            )
        print(
            f"[leaf_correction train-pairs] epoch {epoch + 1}: train {float(t_mse):.5f} val {float(v_mse):.5f}",
            flush=True,
        )
    if best[1] is not None:
        l1.load_state_dict(best[1][0])
        l2.load_state_dict(best[1][1])
        v_mse = torch.as_tensor(best[0])
    w_ = lambda a: a.detach().cpu().numpy().astype(float).round(7).tolist()  # noqa: E731
    net = {
        "version": 1,
        "feature_version": 1,
        "state_dim": int(Z.shape[1]),
        "inputs": inputs,
        "hidden": args.hidden,
        "mean": mean.astype(float).round(7).tolist(),
        "std": std.astype(float).round(7).tolist(),
        "w1": w_(l1.weight),
        "b1": w_(l1.bias),
        "w2": w_(l2.weight[0]),
        "b2": float(l2.bias.item()),
        "clip": args.clip,
        "meta": {
            "pairs": args.pairs,
            "positions": len(pos),
            "val_runs": sorted(val_runs),
            "val_mse": float(v_mse),
            "zero_val_mse": base,
            "anchor": args.anchor,
            "wd": args.wd,
            "objective": "pairwise",
            "best_epoch": best[2],
        },
    }
    Path(args.out).write_text(json.dumps(net), encoding="utf-8")
    print(f"[leaf_correction train-pairs] wrote {args.out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="label decisions with realized - playout value")
    g.add_argument("--starts", required=True, help="JSONL {file, combat_id, turn, mode}")
    g.add_argument("--collect", required=True, help="the `ppo.turnplan.pro collect` dir")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--sims", type=int, default=4000)
    g.add_argument("--k", type=int, default=8, help="belief playouts per decision")
    g.add_argument("--procs", type=int, default=24)
    g.add_argument("--sample-seed", type=int, default=0)
    g.add_argument("--keep-states", action="store_true", help="also write each decision's state")
    t = sub.add_parser("train", help="fit the residual r(s)")
    t.add_argument("--rows", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--hidden", type=int, default=16)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--shrink", type=float, default=0.1, help="penalty on r(s)^2")
    t.add_argument("--clip", type=float, default=0.3, help="|r| cap in value units")
    t.add_argument("--val-frac", type=float, default=0.1)
    t.add_argument("--turn-starts", action="store_true", help="train on turn-start decisions only")
    t.add_argument(
        "--inputs",
        default="x",
        help="comma list of input blocks: x (the 32 rl2 features) and any array of --extra",
    )
    t.add_argument("--extra", help=".npz of row-aligned feature arrays (e.g. K, C)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    tp = sub.add_parser("train-pairs", help="fit r on contrast pairs (pairs_gen + K/C features)")
    tp.add_argument("--pairs", nargs="+", required=True)
    tp.add_argument("--out", required=True)
    tp.add_argument("--inputs", default="x", help="comma list of x, K, C")
    tp.add_argument("--hidden", type=int, default=32)
    tp.add_argument("--epochs", type=int, default=60)
    tp.add_argument("--batch", type=int, default=1024)
    tp.add_argument("--lr", type=float, default=1e-3)
    tp.add_argument("--wd", type=float, default=1e-4)
    tp.add_argument(
        "--anchor", type=float, default=0.05, help="weight of r^2 (keeps the level near 0)"
    )
    tp.add_argument("--clip", type=float, default=0.3)
    tp.add_argument("--val-frac", type=float, default=0.15)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    return {"gen": gen, "train": train, "train-pairs": train_pairs}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
