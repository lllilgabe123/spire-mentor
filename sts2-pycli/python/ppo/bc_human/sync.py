"""Replay a human fight in the engine, decision by decision, checking sync.

The fight is rebuilt on the run's own RNG (`run_seed` + the replay's
`combat_start.rng_state`), so with an exact engine the opening hand, every
shuffle and every monster roll match the game. Each human decision is mapped
to a legal engine action; after it, the engine must agree with the log on the
player's hp, each enemy's hp and the hand the next decision is taken from.
The first disagreement ends the replay: decisions before it are faithful
(state, action) pairs, the rest are not.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import sts2_rs

GAME_VERSION = "v0.111.0"

#: The coach mod, under every name it has shipped as. From 0.7.4 its manifest
#: says `affects_gameplay: false` so a co-op partner can join without it, but
#: its runs are not human play: autoplay plays them and undo rewinds them.
COACH_MOD_IDS = frozenset({"SpireMentor", "SpireCoach", "SimBridge"})


def mods_ok(header: dict[str, Any]) -> bool:
    """No gameplay mod and no coach in a replay header's `mods`."""
    return not any(
        m.get("affects_gameplay") or m.get("id") in COACH_MOD_IDS for m in header.get("mods", [])
    )


# Replay rng_state names -> engine RunRngSet stream names.
_STREAM = {
    "treasure_room_relics": "treasure_room",
    "player_rewards": "rewards",
    "player_shops": "shops",
    "player_transformations": "transformations",
}


def _snake(engine_id: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", engine_id)
    s = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", s)
    return s.upper()


def start_counters(rec: dict[str, Any], monster_ai_offset: int = 0) -> list[tuple[str, int]]:
    """The run's stream counters where the engine's combat setup begins.

    `combat_start.rng_state` is logged after the game has already shuffled
    the draw pile (Fisher-Yates: deck size - 1 reads of `shuffle`) and rolled
    each monster's HP (one `niche` read per monster); rewind both so the
    engine makes those reads itself.
    """
    rng = dict(rec["rng_state"])
    rng["shuffle"] -= max(len(rec["deck"]) - 1, 0)
    rng["niche"] -= len(rec["enemies"])
    rng["monster_ai"] -= monster_ai_offset
    return [(_STREAM.get(k, k), max(int(v), 0)) for k, v in rng.items()]


def build(rec: dict[str, Any], *, roster: bool = True, monster_ai_offset: int = 0) -> Any:
    """The fight's start state; relics the engine does not know are dropped
    (recorded in `rec["dropped_relics"]`) and left for the sync checks."""
    relics = list(rec["relics"])
    rec["dropped_relics"] = []
    while True:
        try:
            return _build(rec, relics, roster, monster_ai_offset)
        except ValueError as e:
            m = re.search(r'unknown relic id "([A-Z0-9_]+)"', str(e))
            if not m or m.group(1) not in relics:
                raise
            relics.remove(m.group(1))
            rec["dropped_relics"].append(m.group(1))


def _build(rec: dict[str, Any], relics: list[str], roster: bool, monster_ai_offset: int) -> Any:
    counters = start_counters(rec, monster_ai_offset)
    belt = [p for p in rec["belt"] if p]
    return sts2_rs.build_scenario_state(
        1,
        rec["character"].capitalize(),
        rec["deck"],
        min(rec["hp"], rec["max_hp"]),
        rec["max_hp"],
        relics,
        rec["encounter"],
        ascension=rec["ascension"] or 0,
        monsters=[e["id"] for e in rec["enemies"]] if roster else None,
        potions=belt,
        potion_slots=max(len(rec["belt"]), len(belt)),
        game_version=GAME_VERSION,
        run_seed=sts2_rs.run_seed_hash(rec["seed"], GAME_VERSION),
        rng_counters=counters,
        # The roster's spawn args (Decimillipede's opener rotation, ...) come
        # from the per-encounter Rng(Seed + TotalFloor + hash(id)); the
        # kind-only `monsters` list carries none of them (GH #918).
        encounter_floor=rec.get("floor"),
    )


class Desync(Exception):
    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def _norm(card_id: str) -> str:
    """Replay and engine card ids differ for a few cards (`CRUELTY` vs `CRUELTY_CARD`)."""
    return card_id.removesuffix("_CARD")


def _hand(st: Any) -> list[tuple[str, bool]]:
    return [(c["id"], bool(c.get("upgraded"))) for c in st.get_state().get("hand", [])]


class _LogView:
    """What the log says the state is, tracked from its events."""

    def __init__(self, rec: dict[str, Any]) -> None:
        self.hp = rec["hp"]
        self.character = rec["character"]
        self.hp_events_carry_damage = any(
            d["t"] == "hp" and d.get("d", 0) < 0 and d.get("dst", "player") == "player"
            for dec in rec["decisions"]
            for d in dec["after"]
        )
        # cid -> (monster id, hp)
        self.enemies: dict[int, list[Any]] = {
            e["cid"]: [e["id"], e["hp"]] for e in rec["enemies"] if "cid" in e
        }
        self.hand: Counter[tuple[str, bool]] = Counter()
        self.ids: dict[int, tuple[str, bool]] = {}  # card c -> (id, upgraded)
        self.belt: list[str | None] = list(rec["belt"])

    def feed(self, d: dict[str, Any]) -> None:
        t = d["t"]
        # From replay_version 6 a `hit`'s `dmg` is the unblocked damage
        # (`blocked` is what block absorbed), and a hit on the player logs
        # no `hp` event; `hp` events (heals, ...) carry the absolute value.
        # Some mod builds also log the hit as a negative `hp` event (whose
        # absolute `hp` can be stale), so deltas are used and a fight with
        # negative `hp` events takes player damage from those alone.
        if t == "hp" and d.get("dst", "player") == "player":
            self.hp += d.get("d", 0)
        elif t == "hit" and d.get("dst") == "player":
            if not self.hp_events_carry_damage:
                self.hp -= d.get("dmg", 0)
        elif t == "hp_loss":
            self.hp -= d.get("dmg", 0)
        elif t == "hit" and d.get("dst_cid") in self.enemies:
            self.enemies[d["dst_cid"]][1] -= d.get("dmg", 0)
        elif t == "spawn" and d.get("side") == "enemy":
            self.enemies[d["cid"]] = [d["id"], d["hp"]]


def _target_index(st: Any, view: _LogView, cid: int | None, name: str | None) -> int | None:
    """Engine enemy slot for the log's target (by cid rank among same-id monsters)."""
    if (name is None and cid is None) or name == view.character:
        return None  # untargeted, or the player themselves
    enemies = st.get_state()["enemies"]
    if cid is not None and cid in view.enemies:
        name = view.enemies[cid][0]
    same = [i for i, e in enumerate(enemies) if _snake(e["id"]) == name]
    if not same:
        raise Desync("target", f"no enemy {name} in engine")
    if len(same) == 1 or cid is None:
        return same[0]
    cids = sorted(c for c, (mid, _) in view.enemies.items() if mid == name)
    rank = cids.index(cid)
    return same[rank] if rank < len(same) else None


def _action(st: Any, view: _LogView, ev: dict[str, Any]) -> dict[str, Any]:
    t = ev["t"]
    if t != "pick" and st.get_state().get("type") == "card_select":
        raise Desync("pick", f"engine awaits a card choice, log has {t}")
    if t == "end_turn":
        return {"action": "end_turn"}
    legal = st.legal_actions()
    if t == "play":
        hand = st.get_state().get("hand", [])
        want = (_norm(ev["id"]), bool(ev.get("up")))
        idxs = [i for i, c in enumerate(hand) if (_norm(c["id"]), bool(c.get("upgraded"))) == want]
        if not idxs:
            raise Desync("hand", f"{want} not in engine hand {_hand(st)}")
        tgt = _target_index(st, view, ev.get("target_cid"), ev.get("target"))
        for a in legal:
            if a.get("action") == "play" and a["card_index"] in idxs:
                if a.get("target_index") is None or tgt is None or a["target_index"] == tgt:
                    return a
        raise Desync("illegal", f"play {want} tgt={tgt} not legal")
    if t in ("potion_start", "potion_used"):
        pots = sts2_rs.state_to_dict(st)["player"].get("potions", [])
        tgt = _target_index(st, view, ev.get("target_cid"), ev.get("target"))
        for a in legal:
            if a.get("action") != "drink_potion" or a["slot"] >= len(pots):
                continue
            if pots[a["slot"]] == ev["id"]:
                if a.get("target_index") is None or tgt is None or a["target_index"] == tgt:
                    return a
        raise Desync("illegal", f"potion {ev['id']} not legal")
    if t == "pick":
        s = st.get_state()
        if s.get("type") != "card_select":
            raise Desync("pick", f"no pending card choice (state {s.get('type')})")
        opts = [(_norm(c["id"]), bool(c.get("upgraded"))) for c in s["cards"]]
        sel: list[int] = []
        for c in ev.get("cards", []):
            want = (_norm(c["id"]), bool(c.get("up")))
            i = next((i for i, o in enumerate(opts) if o == want and i not in sel), None)
            if i is None:
                raise Desync("pick", f"{want} not among options {opts}")
            sel.append(i)
        return {"action": "choose", "index": sel[0] if sel else None, "selection": sel}
    raise Desync("unsupported", t)


def check(st: Any, view: _LogView) -> None:
    s = st.get_state()
    if s.get("type") == "card_select":
        return  # mid-card choice: the next decision is the human's pick
    if "player" not in s:
        raise Desync("state_type", str(s.get("type")))
    if s["player"]["hp"] != view.hp:
        raise Desync("player_hp", f"engine {s['player']['hp']} log {view.hp}")
    eng = Counter((_snake(e["id"]), e["hp"]) for e in s["enemies"] if e["is_alive"])
    log = Counter((mid, hp) for mid, hp in view.enemies.values() if hp > 0)
    if eng != log:
        raise Desync("enemy_hp", f"engine {sorted(eng.items())} log {sorted(log.items())}")


def check_end(decisions: list[dict[str, Any]], k: int, result: Any) -> None:
    """The engine's combat ended at decision `k`; the log's must end there too.

    An engine that wins (or loses) while the log goes on has diverged, though
    every state up to here matched.
    """
    left = len(decisions) - k - 1
    if left:
        raise Desync(
            "ended_early",
            f"engine {result} with {left} log decision(s) left, next {decisions[k + 1]['ev']['t']}",
        )


def replay(rec: dict[str, Any], monster_ai_offset: int | None = None) -> dict[str, Any]:
    """Step the human's decisions; return how far the engine stayed in sync.

    `pairs` holds (state clone, action) for every decision made from a
    state that matched the log.
    """
    out: dict[str, Any] = {"n_dec": len(rec["decisions"]), "n_ok": 0, "desync": None, "pairs": []}
    try:
        off = len(rec["enemies"]) if monster_ai_offset is None else monster_ai_offset
        st = build(rec, monster_ai_offset=off)
    except (ValueError, RuntimeError) as e:
        out["desync"] = ("build", str(e)[:200])
        return out
    view = _LogView(rec)
    for d in rec["pre"]:
        view.feed(d)
    try:
        opening = Counter(_norm(d["id"]) for d in rec["pre"] if d["t"] == "draw")
        eng_open = Counter(_norm(cid) for cid, _ in _hand(st))
        if opening and opening != eng_open:
            raise Desync("opening", f"engine {sorted(eng_open)} log {sorted(opening)}")
        check(st, view)
        decisions = rec["decisions"]
        for k, dec in enumerate(decisions):
            ev = dec["ev"]
            a = _action(st, view, ev)
            out["pairs"].append((st.clone(), a))
            r = st.apply_step(a)
            for d in dec["after"]:
                view.feed(d)
            if r in ("combat_won", "combat_lost") or st.terminal_value()[0]:
                out["n_ok"] = len(out["pairs"])
                check_end(decisions, k, r)
                break
            # The game logs end-of-turn effects (orb passives, Burn/Decay,
            # ...) BEFORE the player's `end_turn` event, so they sit in the
            # previous decision's `after`. Check once the end_turn is applied;
            # a desync there drops the decisions since the last check.
            if k + 1 < len(decisions) and decisions[k + 1]["ev"]["t"] == "end_turn":
                continue
            check(st, view)
            out["n_ok"] = len(out["pairs"])
    except Desync as e:
        out["desync"] = (e.kind, e.detail[:300])
        del out["pairs"][out["n_ok"] :]
    return out
