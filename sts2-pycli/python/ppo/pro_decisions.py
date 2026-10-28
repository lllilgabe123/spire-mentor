"""Run-level decisions of human players, as the PPO run policy sees them.

Replays SpireCodex per-action logs (``replays-spire-codex/<char>/*/*.replay.ndjson.gz``)
and rebuilds, at every card-reward, path and rest decision, the
``(obs, actions, metrics)`` triple the run policy scores (the same shape
:mod:`realgame_ppo` builds from the game bridge), plus the human's choice.
No engine replay is involved: the obs come from the log's own state (deck,
relics, potions, hp, gold, act map), so they hold only what the player saw.

Then scores the champion on those decisions: how often its greedy pick is
the human's, and the probability it gives the human's pick, split by player
strength (`_players.json` A10 win-rate lower bound) and by surface.

    python -m ppo.pro_decisions build --character ironclad --out rows.jsonl
    python -m ppo.pro_decisions score rows.jsonl --out scored.jsonl
    python -m ppo.pro_decisions report scored.jsonl

Surfaces:
- card-pick: every ``card_reward`` decision (pick resolved by the matching
  ``acquire``; no acquire = skip).
- path: every room entered whose parent node had more than one child (the
  options are the parent's children in the act map).
- rest: rest sites where the human chose heal or smith (other options —
  relic-granted — are outside the policy's two-option rest surface).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import sys
from multiprocessing import Pool
from typing import Any

# The scraped SpireCodex replay dump (not in this repo); override with
# SPIRE_CODEX_REPLAYS.
ROOT = os.environ.get(
    "SPIRE_CODEX_REPLAYS",
    os.path.join(
        os.path.expanduser("~"), "projects/sts2/sts2_run_scraper/data/replays-spire-codex"
    ),
)
# Replay `map` node kinds -> the engine's full_map PascalCase room types
# (`mapfeat.nodes_from_full_map` snake-cases them back to the vocab).
KIND = {
    "monster": "Monster",
    "elite": "Elite",
    "restsite": "RestSite",
    "shop": "Shop",
    "treasure": "Treasure",
    "unknown": "Unknown",
    "boss": "Boss",
    "ancient": "Ancient",
}
POTION_SLOTS = 3
START_GOLD = 99


def _coord(s: str) -> tuple[int, int]:
    c, r = s.split(",")
    return int(c), int(r)


class RunState:
    """The player-visible run state, advanced event by event."""

    def __init__(self, header: dict[str, Any]) -> None:
        self.ascension = int(header.get("ascension") or 0)
        self.deck: dict[int, list[Any]] = {
            int(c["c"]): [c["id"], bool(c.get("up"))] for c in header.get("starting_deck") or []
        }
        self.relics: list[str] = list(header.get("starting_relics") or [])
        self.potions: list[str] = []
        self.max_hp = int(header.get("starting_max_hp") or 80)
        self.hp = self.max_hp
        self.gold = START_GOLD
        self.map: dict[tuple[int, int], dict[str, Any]] = {}
        self.boss: dict[str, Any] | None = None
        self.coord: tuple[int, int] | None = None
        self.floor = 0
        self.act = 1
        self.offers: dict[int, list[tuple[str, bool]]] = {}
        self.hp_at_start: int | None = None
        self.map_act: int | None = None

    def feed(self, e: dict[str, Any]) -> None:
        t = e["t"]
        self.floor = int(e.get("floor", self.floor) or self.floor)
        self.act = int(e.get("act", self.act) or self.act)
        if t == "deck":
            self.deck = {int(c["c"]): [c["id"], bool(c.get("up"))] for c in e["cards"]}
        elif t == "acquire":
            up = False
            if e.get("decision_id") in self.offers and e.get("option_index") is not None:
                opts = self.offers[e["decision_id"]]
                if 0 <= e["option_index"] < len(opts):
                    up = opts[e["option_index"]][1]
            self.deck[int(e["c"])] = [e["id"], up]
        elif t == "remove":
            self._take(int(e.get("c", -1)), e.get("id"))
        elif t == "transform":
            # In-combat transforms (e.g. into Giant Rock) hit cards outside the
            # deck: only a deck card's transform changes the deck.
            if self._take(int(e.get("from_c", -1)), e.get("from_id")):
                self.deck[int(e["to_c"])] = [e["to_id"], False]
        elif t == "upgrade":
            c = self._find(int(e.get("c", -1)), e.get("id"), upgraded=False)
            # Offered reward cards are logged as upgrades too; their ids are
            # never in the deck, so a miss is not an error.
            if c is not None and (c == int(e.get("c", -1)) or e.get("decision_id")):
                self.deck[c][1] = True
        elif t == "relic":
            self.relics.append(e["id"])
        elif t == "relic_lost":
            if e.get("id") in self.relics:
                self.relics.remove(e["id"])
        elif t == "potion_got":
            if len(self.potions) < POTION_SLOTS:
                self.potions.append(e["id"])
        elif t in ("potion_used", "potion_dropped"):
            if e.get("id") in self.potions:
                self.potions.remove(e["id"])
        elif t == "hp" and e.get("dst") in (None, "player") and e.get("hp") is not None:
            self.hp = int(e["hp"])
        elif t == "combat_start":
            self.hp_at_start = self.hp
        elif t == "combat_end" and e.get("hp_lost_total") is not None:
            # From replay v6 a hit on the player logs no `hp` event; when none
            # moved hp during the fight, apply the fight's total loss.
            if self.hp == self.hp_at_start:
                self.hp = max(1, self.hp - int(e["hp_lost_total"]))
        elif t == "resume":
            self.hp = int(e.get("hp", self.hp))
            self.gold = int(e.get("gold", self.gold))
        elif t == "max_hp" and e.get("dst") in (None, "player"):
            self.max_hp = int(e["max_hp"])
            if e.get("hp") is not None:
                self.hp = int(e["hp"])
        elif t == "gold" and e.get("gold") is not None:
            self.gold = int(e["gold"])
        elif t == "map":
            self.map = {}
            for n in e["nodes"]:
                self.map[_coord(n["coord"])] = {
                    "kind": n["kind"],
                    "children": [_coord(c) for c in n.get("children") or []],
                }
            bc = _coord(e["boss_coord"]) if e.get("boss_coord") else None
            self.boss = {"col": bc[0], "row": bc[1], "id": e.get("boss")} if bc else None
            # A new act starts at its ancient; a map regenerated mid-act (an
            # event's remap) keeps the player where they stand.
            if int(e.get("act", self.act) or self.act) != self.map_act:
                self.coord = None
            self.map_act = int(e.get("act", self.act) or self.act)
        elif t == "room" and e.get("coord"):
            self.coord = _coord(e["coord"])

    def _find(self, c: int, cid: str | None, upgraded: bool | None = None) -> int | None:
        """The deck instance `c`, or, when a save/resume renumbered the
        instances, the first deck card with the same id (and upgrade state)."""
        if c in self.deck:
            return c
        for k, (kid, up) in self.deck.items():
            if kid == cid and (upgraded is None or up == upgraded):
                return k
        return None

    def _take(self, c: int, cid: str | None) -> bool:
        k = self._find(c, cid)
        if k is None:
            return False
        del self.deck[k]
        return True

    # ── the policy's view ────────────────────────────────────────────────

    def full_map(self) -> dict[str, Any] | None:
        if not self.map:
            return None
        rows: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for (c, r), n in sorted(self.map.items()):
            if n["kind"] in ("boss", "ancient"):
                continue
            rows[r].append(
                {
                    "col": c,
                    "row": r,
                    "type": KIND.get(n["kind"], "Unknown"),
                    "children": [
                        {"col": cc, "row": rr}
                        for cc, rr in n["children"]
                        if self.map.get((cc, rr), {}).get("kind") not in ("boss",)
                    ],
                }
            )
        return {"rows": [rows[r] for r in sorted(rows)], "boss": self.boss}

    def map_coord(self) -> tuple[int, int] | None:
        # The engine reports no coord at an act's first choice; the log's
        # pre-act ancient node sits on row 0, which the act graph never holds.
        if self.coord is None or self.coord[1] == 0:
            return None
        return self.coord

    def obs(self, surface: str) -> dict[str, Any]:
        from realgame_ppo import _vocab_card_id

        return {
            "phase": None,
            "surface": surface,
            "seed": 0,
            "ascension": self.ascension,
            "deck": [(_vocab_card_id(cid), up) for cid, up in self.deck.values()],
            "relics": list(self.relics),
            "potions": list(self.potions),
            "map": self.full_map(),
            "map_coord": self.map_coord(),
            "max_hp": self.max_hp,
        }

    def metrics(self) -> dict[str, Any]:
        mc = self.map_coord()
        return {
            "ascension": self.ascension,
            "total_floor": max(0, self.floor - 1),
            "act_index": self.act,
            "act_floor": mc[1] + 1 if mc else 0,
            "hp": self.hp,
            "max_hp": self.max_hp,
            "gold": self.gold,
            "deck_size": len(self.deck),
            "num_upgraded": sum(1 for _, up in self.deck.values() if up),
            "num_relics": len(self.relics),
            "num_potions": len(self.potions),
        }


def parse(path: str, character: str) -> dict[str, Any]:
    """Every scorable decision of one replay, or a reason it was skipped."""
    from realgame_ppo import _vocab_card_id

    events = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    head = events[0]
    meta = {"file": os.path.basename(path).split(".")[0], "build": head.get("build_id")}
    if head.get("t") != "header" or int(head.get("floor") or 0) != 0:
        return meta | {"skip": "starts_mid_run"}
    if str(head.get("character", "")).upper() != character.upper():
        return meta | {"skip": "character"}
    if int(head.get("player_count") or 1) != 1 or any(e.get("mine") is False for e in events):
        return meta | {"skip": "multiplayer"}
    if head.get("game_mode") != "standard" or head.get("modifiers"):
        return meta | {"skip": "mode"}
    if any(m.get("affects_gameplay") for m in head.get("mods") or []):
        return meta | {"skip": "mod"}
    st = RunState(head)
    rows: list[dict[str, Any]] = []
    pending: dict[int, dict[str, Any]] = {}  # card_reward decision_id -> row
    rest_row: dict[str, Any] | None = None
    rest_at: tuple[dict[str, Any], dict[str, Any]] | None = None
    final_deck = None
    for e in events[1:]:
        t = e["t"]
        if t == "decision" and e.get("decision_type") == "card_reward":
            opts = [(o["option_id"], bool(o.get("up"))) for o in e.get("options") or []]
            st.offers[e["decision_id"]] = opts
            obs = st.obs("card-pick")
            obs["card_reward_option_ids"] = [_vocab_card_id(o[0]) for o in opts]
            obs["card_reward_option_upgraded"] = [o[1] for o in opts]
            actions = [{"action": "pick_card_reward", "idx": i} for i in range(len(opts))]
            if e.get("decline_available", True):
                actions.append({"action": "pick_card_reward", "idx": None})
            row = {
                "surface": "card-pick",
                "obs": obs,
                "actions": actions,
                "metrics": st.metrics(),
                "human": len(opts) if e.get("decline_available", True) else None,
                "labels": [o[0] for o in opts] + ["SKIP"],
            }
            pending[e["decision_id"]] = row
            rows.append(row)
        elif (
            t == "acquire" and e.get("decision_id") in pending and e.get("option_index") is not None
        ):
            pending.pop(e["decision_id"])["human"] = int(e["option_index"])
        elif t == "room" and e.get("coord"):
            new = _coord(e["coord"])
            prev = st.coord
            parent = st.map.get(prev) if prev is not None else None
            if prev is None and st.map:
                # first room of the act: the options are the act's row-1 nodes
                kids = sorted(
                    c for c in st.map if c[1] == new[1] and st.map[c]["kind"] != "ancient"
                )
            else:
                kids = list(parent["children"]) if parent else []
            if new[1] > 0 and new in kids and len(kids) > 1:
                rows.append(
                    {
                        "surface": "path",
                        "obs": st.obs("path"),
                        "actions": [{"action": "move", "col": c} for c, _ in kids],
                        "metrics": st.metrics(),
                        "human": kids.index(new),
                        "labels": [st.map[k]["kind"] for k in kids],
                    }
                )
        elif t == "rest" and e.get("option") in ("heal", "smith") and rest_at is not None:
            # The heal's `hp` event and the smith's `upgrade` are logged
            # BEFORE `rest`: decide on the state at the rest site's entry.
            obs, metrics = rest_at
            obs["rest_option_labels"] = ["rest", "smith"]
            rest_row = {
                "surface": "rest",
                "obs": obs,
                "actions": [{"action": "rest_option", "idx": i} for i in range(2)],
                "metrics": metrics,
                "human": 0 if e["option"] == "heal" else 1,
                "labels": ["heal", "smith"],
            }
            rows.append(rest_row)
        elif t == "end":
            final_deck = e.get("final_deck")
        st.feed(e)
        if t == "room":
            rest_at = (st.obs("rest"), st.metrics()) if e.get("kind") == "restsite" else None
    # Unresolved card rewards with no decline option were never answered.
    rows = [r for r in rows if r["human"] is not None]
    deck_ok = None
    if final_deck is not None:
        want = collections.Counter(c["id"] for c in final_deck)
        deck_ok = want == collections.Counter(cid for cid, _ in st.deck.values())
    return meta | {
        "ascension": st.ascension,
        "deck_ok": deck_ok,
        "rows": rows,
    }


def _load_players(min_runs: int) -> dict[str, dict[str, Any]]:
    """run_hash -> {player_token, lo95, win} from the listing + players file."""
    players = {
        p["player_token"]: p
        for p in json.load(open(f"{ROOT}/_players.json", encoding="utf-8"))
        if p["a10_runs"] >= min_runs
    }
    out = {}
    for line in open(f"{ROOT}/_listing.jsonl", encoding="utf-8"):
        r = json.loads(line)
        p = players.get(r.get("player_token"))
        out[r["run_hash"]] = {
            "player": r.get("player_token"),
            "lo95": p["a10_win_rate_lo95"] if p else None,
            "win": bool(r.get("win")),
        }
    return out


def _parse_job(job: tuple[str, str]) -> dict[str, Any]:
    try:
        return parse(*job)
    except Exception as ex:  # one bad log never stops the corpus
        return {"file": os.path.basename(job[0]).split(".")[0], "skip": f"error {ex!r}"[:200]}


def build(args: argparse.Namespace) -> int:
    import glob

    files = sorted(glob.glob(f"{ROOT}/{args.character.lower()}/*/*.replay.ndjson.gz"))
    if args.limit:
        files = files[: args.limit]
    meta = _load_players(args.min_a10_runs)
    skips: collections.Counter[str] = collections.Counter()
    n_rows = collections.Counter()
    deck_ok = collections.Counter()
    with Pool(args.procs) as pool, open(args.out, "w", encoding="utf-8") as out:
        for res in pool.imap_unordered(
            _parse_job, [(f, args.character) for f in files], chunksize=8
        ):
            if "skip" in res:
                skips[res["skip"].split(" ")[0]] += 1
                continue
            deck_ok[res["deck_ok"]] += 1
            m = meta.get(res["file"], {})
            for r in res["rows"]:
                r.update(
                    file=res["file"],
                    build=res["build"],
                    ascension=res["ascension"],
                    player=m.get("player"),
                    lo95=m.get("lo95"),
                    win=m.get("win"),
                )
                out.write(json.dumps(r) + "\n")
                n_rows[r["surface"]] += 1
    print(f"[pro_decisions] {len(files)} replays; skipped {dict(skips)}")
    print(f"[pro_decisions] final deck reconstructed exactly: {dict(deck_ok)}")
    print(f"[pro_decisions] rows {dict(n_rows)}")
    return 0


_POLICY: Any = None


def _score_init(ckpt: str) -> None:
    global _POLICY
    import torch

    torch.set_num_threads(1)
    from realgame_ppo import _load_torch

    _POLICY, err = _load_torch(ckpt)
    if _POLICY is None:
        raise SystemExit(f"cannot load {ckpt}: {err}")


def _score_row(line: str) -> str:
    r = json.loads(line)
    try:
        probs, value = _POLICY.distribution(r["obs"], r["actions"], r["metrics"])
    except Exception as ex:
        return json.dumps({"error": repr(ex)[:200], "surface": r["surface"]})
    h = r["human"]
    out = {
        k: r[k]
        for k in (
            "surface",
            "file",
            "build",
            "ascension",
            "player",
            "lo95",
            "win",
            "labels",
            "human",
        )
    }
    out.update(
        probs=[round(p, 5) for p in probs],
        value=round(value, 4),
        top=max(range(len(probs)), key=probs.__getitem__),
        p_human=probs[h],
        act=r["metrics"]["act_index"],
        floor=r["metrics"]["total_floor"],
    )
    return json.dumps(out)


def score(args: argparse.Namespace) -> int:
    ckpt = args.ckpt
    if not ckpt:
        from workbench.ppo_policy import ckpt_path

        ckpt = ckpt_path(args.character)
    print(f"[pro_decisions] scoring with {ckpt}", flush=True)
    lines = open(args.rows, encoding="utf-8").readlines()
    n_err = 0
    with (
        Pool(args.procs, initializer=_score_init, initargs=(ckpt,)) as pool,
        open(args.out, "w", encoding="utf-8") as out,
    ):
        for i, s in enumerate(pool.imap(_score_row, lines, chunksize=64)):
            n_err += '"error"' in s[:12]
            out.write(s + "\n")
            if (i + 1) % 20000 == 0:
                print(f"[pro_decisions] {i + 1}/{len(lines)} scored, {n_err} errors", flush=True)
    print(f"[pro_decisions] {len(lines)} scored, {n_err} errors -> {args.out}")
    return 0


def _tier(r: dict[str, Any], pro_cut: float) -> str:
    if r.get("lo95") is None:
        return "other"
    return "pro" if r["lo95"] >= pro_cut else "rated"


def report(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in open(args.scored, encoding="utf-8")]
    rows = [r for r in rows if "error" not in r]
    if args.min_ascension:
        rows = [r for r in rows if (r.get("ascension") or 0) >= args.min_ascension]
    los = sorted({r["lo95"] for r in rows if r.get("lo95") is not None}, reverse=True)
    players = {}
    for r in rows:
        if r.get("lo95") is not None:
            players[r["player"]] = r["lo95"]
    ranked = sorted(players.values(), reverse=True)
    pro_cut = ranked[min(args.top - 1, len(ranked) - 1)] if ranked else math.inf
    print(
        f"rows {len(rows)} (asc >= {args.min_ascension}); rated players {len(players)}; "
        f"pro = top {args.top} by A10 win-rate lo95 (cut {pro_cut:.3f}); {len(los)} distinct lo95"
    )

    def line(name: str, rs: list[dict[str, Any]]) -> None:
        if not rs:
            return
        n = len(rs)
        agree = sum(r["top"] == r["human"] for r in rs) / n
        ph = sum(r["p_human"] for r in rs) / n
        nll = -sum(math.log(max(r["p_human"], 1e-6)) for r in rs) / n
        chance = sum(1 / len(r["probs"]) for r in rs) / n
        print(
            f"  {name:34s} n {n:6d}  agree {100 * agree:5.1f}%  p(human) {ph:.3f}  nll {nll:.3f}  chance {100 * chance:4.1f}%"
        )

    for surface in ("card-pick", "path", "rest"):
        srows = [r for r in rows if r["surface"] == surface]
        print(f"\n== {surface}")
        for tier in ("pro", "rated", "other"):
            line(tier, [r for r in srows if _tier(r, pro_cut) == tier])
        line("pro, won runs", [r for r in srows if _tier(r, pro_cut) == "pro" and r.get("win")])
        for act in (1, 2, 3):
            line(
                f"pro, act {act}",
                [r for r in srows if _tier(r, pro_cut) == "pro" and r["act"] == act],
            )
        if surface == "card-pick":
            for tier in ("pro", "rated", "other"):
                tr = [r for r in srows if _tier(r, pro_cut) == tier]
                if tr:
                    hs = sum(r["labels"][r["human"]] == "SKIP" for r in tr) / len(tr)
                    bs = sum(r["labels"][r["top"]] == "SKIP" for r in tr) / len(tr)
                    print(
                        f"  skip rate {tier:6s}: human {100 * hs:5.1f}%  champion {100 * bs:5.1f}%"
                    )
        if surface == "rest":
            for tier in ("pro", "rated"):
                tr = [r for r in srows if _tier(r, pro_cut) == tier]
                if tr:
                    hs = sum(r["human"] == 1 for r in tr) / len(tr)
                    bs = sum(r["top"] == 1 for r in tr) / len(tr)
                    print(
                        f"  smith rate {tier:6s}: human {100 * hs:5.1f}%  champion {100 * bs:5.1f}%"
                    )
    if args.cards:
        pro = [r for r in rows if r["surface"] == "card-pick" and _tier(r, pro_cut) == "pro"]
        offered: collections.Counter[str] = collections.Counter()
        took_h: collections.Counter[str] = collections.Counter()
        took_b: collections.Counter[str] = collections.Counter()
        for r in pro:
            for i, lab in enumerate(r["labels"][:-1]):
                offered[lab] += 1
                took_h[lab] += r["human"] == i
                took_b[lab] += r["top"] == i
        diff = [
            (took_h[c] / offered[c] - took_b[c] / offered[c], c)
            for c in offered
            if offered[c] >= args.cards
        ]
        diff.sort()
        print(f"\n== pro card picks: largest gaps in take rate (offered >= {args.cards})")
        print("  champion takes more:")
        for d, c in diff[:12]:
            print(
                f"    {c:24s} offered {offered[c]:4d}  pro {100 * took_h[c] / offered[c]:5.1f}%  champion {100 * took_b[c] / offered[c]:5.1f}%"
            )
        print("  pros take more:")
        for d, c in diff[::-1][:12]:
            print(
                f"    {c:24s} offered {offered[c]:4d}  pro {100 * took_h[c] / offered[c]:5.1f}%  champion {100 * took_b[c] / offered[c]:5.1f}%"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--character", default="ironclad")
    b.add_argument("--out", required=True)
    b.add_argument("--procs", type=int, default=16)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--min-a10-runs", type=int, default=15)
    s = sub.add_parser("score")
    s.add_argument("rows")
    s.add_argument("--out", required=True)
    s.add_argument("--character", default="ironclad")
    s.add_argument("--ckpt", default="")
    s.add_argument("--procs", type=int, default=16)
    r = sub.add_parser("report")
    r.add_argument("scored")
    r.add_argument("--top", type=int, default=25)
    r.add_argument("--min-ascension", type=int, default=0)
    r.add_argument("--cards", type=int, default=0, help="per-card take-rate gaps, min offers")
    args = ap.parse_args(argv)
    return {"build": build, "score": score, "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
